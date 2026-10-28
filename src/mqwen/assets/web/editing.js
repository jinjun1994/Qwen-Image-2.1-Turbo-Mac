window.mqwenEditingReady=(async()=>{
await window.mqwenLocaleReady;
const $=id=>document.getElementById(id),t=(key,values)=>window.mqwenLocale.t(key,values);
let refs=[],uploading=false,uploadedMask=null,painted=false,painting=false;
const canvas=$('mask-canvas'),ctx=canvas.getContext('2d');
async function upload(file){
 if(!file||file.size>20*1024*1024)throw Error(t('Upload an image up to 20 MiB'));
 const response=await fetch('/api/uploads',{method:'POST',headers:{'Content-Type':file.type,'Accept-Language':window.mqwenLocale.language},body:file});
 const data=await response.json();if(!response.ok)throw Error(data.error||t('Request failed'));return data;
}
function error(message){$('error').textContent=message;$('error').hidden=!message;}
function resetMask(){uploadedMask=null;painted=false;painting=false;ctx.clearRect(0,0,canvas.width,canvas.height);$('mask-file').value='';maskStatus();}
function maskStatus(){$('mask-status').textContent=t(painted?'Painted mask active':uploadedMask?'Uploaded mask active':'No mask: edit the whole image');}
function render(){
 $('reference-list').replaceChildren();refs.forEach((ref,index)=>{
  const card=document.createElement('div');card.className='reference-card';
  const img=document.createElement('img');img.src=ref.url;img.alt=t('Image {number}',{number:index+1});
  const label=document.createElement('span');label.textContent=`${index+1} · ${ref.width} × ${ref.height}`;
  const remove=document.createElement('button');remove.type='button';remove.textContent=t('Remove');remove.onclick=()=>{const first=index===0;refs.splice(index,1);if(first)resetMask();render();};
  card.append(img,label,remove);$('reference-list').append(card);
 });
 $('mask-stage').hidden=!refs.length;
 if(refs.length&&$('mask-source').getAttribute('src')!==refs[0].url){
  $('mask-source').src=refs[0].url;const scale=Math.min(1,1024/Math.max(refs[0].width,refs[0].height));canvas.width=Math.round(refs[0].width*scale);canvas.height=Math.round(refs[0].height*scale);resetMask();
 }
 maskStatus();
}
function taskChanged(){
 const editing=$('task').value!=='generate';$('reference-panel').hidden=!editing;
 $('step-cache').disabled=editing||$('steps').value!=='8';if(editing)$('step-cache').checked=false;
 if($('task').value==='background'){$('output-format').value='rgba';$('prompt').value='Remove the background from <image1>, preserve the subject and its details, and output a PNG image with a transparent background.';}
 $('channel-label').textContent=$('output-format').value.toUpperCase()+' / PNG';
}
$('task').onchange=taskChanged;$('output-format').onchange=taskChanged;
$('reference-files').onchange=async()=>{
 const files=Array.from($('reference-files').files);if(refs.length+files.length>10){error(t('Use at most 10 reference images'));return;}
 uploading=true;$('reference-files').disabled=true;
 try{for(const file of files)refs.push(await upload(file));error('');}catch(exc){error(exc.message);}finally{uploading=false;$('reference-files').disabled=false;$('reference-files').value='';render();}
};
$('mask-file').onchange=async()=>{
 if(!refs.length){error(t('A mask requires a reference image'));return;}
 uploading=true;try{uploadedMask=await upload($('mask-file').files[0]);painted=false;ctx.clearRect(0,0,canvas.width,canvas.height);error('');maskStatus();}catch(exc){error(exc.message);}finally{uploading=false;}
};
$('mask-clear').onclick=resetMask;
function point(event){const bounds=canvas.getBoundingClientRect();return [(event.clientX-bounds.left)*canvas.width/bounds.width,(event.clientY-bounds.top)*canvas.height/bounds.height];}
function stroke(event){
 if(!painting)return;const [x,y]=point(event);ctx.lineWidth=Number($('brush-size').value)*canvas.width/canvas.getBoundingClientRect().width;ctx.lineCap='round';ctx.lineJoin='round';ctx.strokeStyle='white';ctx.lineTo(x,y);ctx.stroke();ctx.beginPath();ctx.moveTo(x,y);painted=true;uploadedMask=null;maskStatus();
}
canvas.onpointerdown=event=>{event.preventDefault();canvas.setPointerCapture(event.pointerId);painting=true;ctx.beginPath();const [x,y]=point(event);ctx.moveTo(x,y);ctx.lineTo(x+.01,y+.01);stroke(event);};
canvas.onpointermove=stroke;canvas.onpointerup=canvas.onpointercancel=()=>{painting=false;ctx.beginPath();};
for(const [id,instruction] of [
 ['preserve-person','Preserve the identity, facial features, hairstyle and distinctive details of the person in <image1>.'],
 ['preserve-product','Preserve the product shape, proportions, materials, logo and packaging details in <image1>.']
])$(id).onclick=()=>{$('prompt').value=$('prompt').value.trim()+'\n'+instruction;};
window.addEventListener('mqwen-language-change',render);
window.mqwenEditing={payload:async()=>{
 if(uploading)throw Error(t('Wait for image uploads to finish'));
 const editing=$('task').value!=='generate';if(editing&&!refs.length)throw Error(t('Upload at least one source image'));
 let mask=uploadedMask?.id||null;
 if(editing&&painted){
  const output=document.createElement('canvas');output.width=canvas.width;output.height=canvas.height;const c=output.getContext('2d');c.fillStyle='black';c.fillRect(0,0,output.width,output.height);c.drawImage(canvas,0,0);
  const blob=await new Promise(resolve=>output.toBlob(resolve,'image/png'));mask=(await upload(blob)).id;
 }
 return {references:editing?refs.map(ref=>ref.id):[],mask:editing?mask:null,output_format:$('output-format').value,step_cache:editing?false:$('step-cache').checked};
}};
taskChanged();render();
})();
