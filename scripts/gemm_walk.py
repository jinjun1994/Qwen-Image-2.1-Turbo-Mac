"""Experimental output-tile schedules; matrix arithmetic uses the existing kernel."""
from functools import lru_cache
from pathlib import Path

SCHEDULES=('row','column','group4','group8','group16','morton2','morton4','morton8','morton16','morton')


def launch_spec(rows,columns,schedule):
    if rows<1 or columns<1:raise ValueError('Tile counts must be positive')
    if schedule not in SCHEDULES:raise ValueError('Unknown GEMM walk order')
    mb=(rows-1).bit_length();nb=(columns-1).bit_length()
    group=1
    if schedule=='row':mode=0;count=rows*columns
    elif schedule=='column':mode=1;count=rows*columns
    elif schedule.startswith('group'):
        mode=2;group=int(schedule[5:]);count=rows*columns
    elif schedule=='morton':mode=4;count=(1<<mb)*(1<<nb)
    else:
        mode=3;group=int(schedule[6:]);count=((rows+group-1)//group)*((columns+group-1)//group)*group*group
    return mode,group,mb,nb,count


def coordinate(index,rows,columns,schedule):
    mode,group,mb,nb,count=launch_spec(rows,columns,schedule)
    if not 0<=index<count:raise ValueError('Threadgroup index out of range')
    if mode==0:row,column=divmod(index,columns)
    elif mode==1:column,row=divmod(index,rows)
    elif mode==2:
        block,local=divmod(index,group*columns)
        first=block*group;actual=min(group,rows-first)
        column,offset=divmod(local,actual);row=first+offset
    elif mode==3:
        block,local=divmod(index,group*group)
        blockrow,blockcol=divmod(block,(columns+group-1)//group)
        row=blockrow*group;column=blockcol*group
        for bit in range((group-1).bit_length()):
            column|=((local>>(2*bit))&1)<<bit
            row|=((local>>(2*bit+1))&1)<<bit
    else:
        row=column=0;remaining=index
        for bit in range(max(mb,nb)):
            if bit<nb:column|=(remaining&1)<<bit;remaining>>=1
            if bit<mb:row|=(remaining&1)<<bit;remaining>>=1
    return (row,column) if row<rows and column<columns else None


@lru_cache(maxsize=1)
def kernel():
    import mlx.core as mx
    import mqwen
    header=(Path(mqwen.__file__).parent/'assets/kernels/int8_tensorwise.metal').read_text()
    return mx.fast.metal_kernel(name='mqwen_output_walk_qmm',input_names=['x','w','s','b'],output_names=['y'],header=header,source='''
    constexpr uint MT=(M+63)/64,NT=N/64;
    uint p=threadgroup_position_in_grid.x;
    uint row=0,col=0;
    if constexpr(MODE==0){row=p/NT;col=p%NT;}
    else if constexpr(MODE==1){row=p%MT;col=p/MT;}
    else if constexpr(MODE==2){
        uint block=p/(GROUP*NT),local=p%(GROUP*NT);
        uint first=block*GROUP,actual=min(uint(GROUP),MT-first);
        row=first+local%actual;col=local/actual;
    }else if constexpr(MODE==3){
        uint block=p/(GROUP*GROUP),local=p%(GROUP*GROUP);
        row=(block/((NT+GROUP-1)/GROUP))*GROUP;
        col=(block%((NT+GROUP-1)/GROUP))*GROUP;
        STEEL_PRAGMA_UNROLL
        for(uint bit=0;(1u<<bit)<GROUP;bit++){
            col|=((local>>(2*bit))&1u)<<bit;
            row|=((local>>(2*bit+1))&1u)<<bit;
        }
    }else{
        STEEL_PRAGMA_UNROLL
        for(uint bit=0;bit<max(MBITS,NBITS);bit++){
            if(bit<NBITS){col|=(p&1u)<<bit;p>>=1;}
            if(bit<MBITS){row|=(p&1u)<<bit;p>>=1;}
        }
    }
    if(row>=MT || col>=NT)return;
    threadgroup T Xs[64*40];threadgroup T Ws[64*40];
    qmm_t_impl<T,64,8,true,T,64,32,64>(w,s,b,x,y,Xs,Ws,K,N,M,K,
        uint3(col,row,0),thread_index_in_threadgroup,
        simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
    ''')


def matmul(x,layer,schedule):
    import mlx.core as mx
    if x.ndim<2 or x.dtype!=mx.bfloat16 or x.shape[-1]!=layer.in_features:
        raise ValueError('Expected BF16 matrix input with matching K')
    k=x.shape[-1];n=layer.out_features
    if k<1 or k%64 or n<1 or n%64:raise ValueError('K/N must be positive multiples of 64')
    m=x.size//k
    mode,group,mb,nb,count=launch_spec((m+63)//64,n//64,schedule)
    return kernel()(inputs=[x,layer.weight,layer.scales[:,:1].reshape(-1),layer.biases[:,:1].reshape(-1)],
        template=[('T',x.dtype),('M',m),('N',n),('K',k),('MODE',mode),('GROUP',group),('MBITS',mb),('NBITS',nb)],
        grid=(count*128,1,1),threadgroup=(128,1,1),output_shapes=[(*x.shape[:-1],n)],output_dtypes=[x.dtype])[0]
