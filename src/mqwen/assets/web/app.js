(async()=>{
await window.mqwenLocaleReady;
const $=id=>document.getElementById(id);
const t=(key,values)=>window.mqwenLocale.t(key,values);
const headers=()=>({'Accept-Language':window.mqwenLocale.language});
let engineReady=false,connectionError=false;
let mode='resolution',busy=false,selected=null,lastState=null,gallerySignature='';
let modelSettings=null,downloadCatalog=null,downloadTask=null,downloadTimer=null;
const pathInputs={};
let deleteTarget=null,deleteBusy=false;
function size(){
  if(mode==='resolution')return [Number($('width').value),Number($('height').value)];
  const [a,b]=$('ratio').value.split(':').map(Number),area=Number($('megapixels').value)*1e6;
  return [Math.max(32,Math.round(Math.sqrt(area*a/b)/32)*32),Math.max(32,Math.round(Math.sqrt(area*b/a)/32)*32)];
}
function updateSize(){
  const [w,h]=size();$('mp-label').textContent=Number($('megapixels').value).toFixed(2)+' MP';$('size-label').textContent=`${w} × ${h} px`;$('actual-mp').textContent=(w*h/1e6).toFixed(2)+' MP';
  const valid=[w,h].every(n=>Number.isInteger(n)&&n>=256&&n<=2048&&n%32===0);
  $('size-note').textContent=t(valid?'Dimensions are aligned to 32 px; actual pixels may differ slightly from the target.':'Dimensions must be multiples of 32 between 256 and 2048. Reduce pixels or adjust the aspect ratio.');
  $('size-note').classList.toggle('invalid',!valid);$('generate').disabled=busy||!valid||!engineReady;return valid;
}
function setMode(next){mode=next;$('pixels-fields').hidden=mode!=='pixels';$('resolution-fields').hidden=mode!=='resolution';$('pixels-mode').setAttribute('aria-pressed',mode==='pixels');$('resolution-mode').setAttribute('aria-pressed',mode==='resolution');$('width').required=$('height').required=mode==='resolution';updateSize();}
async function jsonGet(path){const r=await fetch(path,{headers:headers()});const data=await r.json();if(!r.ok)throw Error(data.error||t('Request failed'));return data;}
async function post(path,data){const r=await fetch(path,{method:'POST',headers:{...headers(),'Content-Type':'application/json'},body:JSON.stringify(data)}),body=await r.json();if(!r.ok)throw Error(body.error||t('Request failed'));return body;}
function showError(text){$('error').textContent=text;$('error').hidden=!text;}
function message(text){$('settings-message').textContent=text;}
function updateSteps(){
  $('step-cache').disabled=$('steps').value!=='8';if($('steps').value!=='8')$('step-cache').checked=false;
  $('steps-note').textContent=t($('steps').value==='8'?'Standard mode uses the original 8-step schedule.':'Reducing steps changes sampling and image quality; 4 steps are for previews.');
}
$('pixels-mode').onclick=()=>setMode('pixels');$('resolution-mode').onclick=()=>setMode('resolution');
for(const id of ['megapixels','ratio','width','height'])$(id).addEventListener('input',updateSize);
$('swap').onclick=()=>{const w=$('width').value;$('width').value=$('height').value;$('height').value=w;updateSize();};
document.querySelectorAll('[data-size]').forEach(b=>b.onclick=()=>{[$('width').value,$('height').value]=b.dataset.size.split(',');updateSize();});
$('random').onclick=()=>{$('seed').value='';};$('steps').onchange=updateSteps;
let promptPresets=null;
$('random-prompt').onclick=async()=>{
  $('random-prompt').disabled=true;
  try{
    if(!promptPresets){const data=await jsonGet('/prompts.json');if(!Array.isArray(data)||data.length!==6||data.some(value=>typeof value!=='string'||!value.trim()))throw Error(t('Prompt presets are unavailable'));promptPresets=data;}
    $('prompt').value=promptPresets[Math.floor(Math.random()*promptPresets.length)];
    $('prompt').dispatchEvent(new Event('input',{bubbles:true}));showError('');
  }catch(error){showError(error.message);}
  finally{$('random-prompt').disabled=false;}
};
$('generation').onsubmit=async event=>{
  event.preventDefault();if(!updateSize()||busy||!engineReady)return;showError('');
  const payload={mode,steps:Number($('steps').value),step_cache:$('step-cache').checked,prompt:$('prompt').value,seed:$('seed').value,megapixels:Number($('megapixels').value),ratio:$('ratio').value,width:Number($('width').value),height:Number($('height').value)};
  busy=true;updateSize();try{selected=(await post('/api/generate',payload)).id;await refresh();}catch(error){busy=false;updateSize();showError(error.message);}
};
$('cancel').onclick=async()=>{try{await post('/api/cancel',{});await refresh();}catch(error){showError(error.message);}};
for(const operation of ['start','stop'])$('engine-'+operation).onclick=async()=>{showError('');try{await post('/api/engine/'+operation,{});await refresh();}catch(error){showError(error.message);}};
function render(state){
  $('settings-open').hidden=!state.desktop;engineReady=state.engine_state==='ready';
  $('engine-start').disabled=!['stopped','failed'].includes(state.engine_state);$('engine-stop').disabled=['stopped','stopping','failed'].includes(state.engine_state);
  $('engine-state').textContent=t(({ready:'Models resident · Ready',loading:'Loading models',stopping:'Releasing models',stopped:'Inference stopped',failed:'Model loading failed'})[state.engine_state]||state.engine_state);
  if(state.engine_error)showError(state.engine_error);lastState=state;busy=!!state.active;updateSize();$('cancel').hidden=!busy;
  $('cancel').disabled=state.jobs.some(j=>j.id===state.active&&j.status==='cancelling');
  if(state.active)selected=state.active;
  if(!state.jobs.some(j=>j.id===selected))selected=state.jobs.find(j=>j.status==='completed')?.id||state.jobs[0]?.id;
  const job=state.jobs.find(j=>j.id===selected);$('delete-result').hidden=!job||job.status!=='completed';$('delete-result').disabled=deleteBusy;$('image-stage').classList.toggle('busy',busy);
  if(job){
    $('status').textContent=job.stage;
    $('result-info').textContent=`${job.params.width} × ${job.params.height} · ${t('Seed')} ${job.params.seed} · ${t('{count} steps',{count:job.params.steps||8})}`+(job.params.step_cache?' · '+t('Cross-step cache'):'')+(job.total_seconds?' · '+t('{count}s',{count:job.total_seconds.toFixed(1)}):'');
    $('progress').hidden=!['running','cancelling'].includes(job.status);$('progress').value=job.progress;$('logs').textContent=job.logs.join('\n')||t('Saved generation result');
    if(job.status==='failed')showError(job.error||t('Generation failed; check the logs'));
    if(job.image_url){$('result-image').hidden=false;$('empty').hidden=true;if($('result-image').getAttribute('src')!==job.image_url)$('result-image').src=job.image_url;$('result-image').alt=job.params.prompt;$('download').hidden=false;$('download').href=job.image_url;}
    else{$('result-image').hidden=true;$('empty').hidden=false;$('download').hidden=true;}
  }
  const completed=state.jobs.filter(j=>j.status==='completed'),signature=completed.map(j=>j.id).join(',')+'|'+selected+'|'+window.mqwenLocale.language;
  if(signature!==gallerySignature){
    gallerySignature=signature;const gallery=$('gallery');gallery.replaceChildren();
    for(const j of completed){const button=document.createElement('button');button.type='button';button.className='thumbnail'+(j.id===selected?' selected':'');button.title=j.params.prompt;
      const image=document.createElement('img');image.src=j.image_url;image.alt=j.params.prompt;image.loading='lazy';const label=document.createElement('span');label.textContent=`${j.params.width} × ${j.params.height} · ${j.params.seed}`;
      button.append(image,label);button.onclick=()=>{if(busy)return;selected=j.id;showError('');render(lastState);};
      const card=document.createElement('div');card.className='gallery-card';const remove=document.createElement('button');remove.type='button';remove.className='delete-image';remove.textContent=t('Move to Trash');remove.setAttribute('aria-label',t('Delete image')+' '+j.params.seed);remove.disabled=deleteBusy;remove.onclick=()=>confirmDelete(j.id);card.append(button,remove);gallery.append(card);}
    if(!completed.length){const p=document.createElement('p');p.className='hint';p.textContent=t('No saved images yet.');gallery.append(p);}
  }
}
async function refresh(){
  const locale=window.mqwenLocale.language;
  try{const state=await jsonGet('/api/state');if(locale!==window.mqwenLocale.language)return;
    if(connectionError){showError('');connectionError=false;}render(state);
  }catch(error){connectionError=true;$('status').textContent=t('Connection lost');showError(error.message+' '+t('Check that the WebUI service is still running.'));}
}
function pick(key){if(window.webkit?.messageHandlers?.mqwen)window.webkit.messageHandlers.mqwen.postMessage({key});else message(t('Paste an absolute path in the browser. The desktop app supports file selection.'));}
window.mqwenPicked=({key,path})=>{if(key==='download_dir')$('download-dir').value=path;else if(pathInputs[key])pathInputs[key].value=path;};
document.querySelectorAll('[data-pick]').forEach(button=>button.onclick=()=>pick(button.dataset.pick));
async function loadSettings(){
  try{modelSettings=await jsonGet('/api/settings');const box=$('model-paths');box.replaceChildren();
    for(const [role,key] of Object.entries({dit:'Turbo DiT',text_encoder:'Text encoder',vae:'VAE'})){
      const label=document.createElement('label');label.dataset.i18n=key;label.textContent=t(key);
      const row=document.createElement('div');row.className='path-row';const input=document.createElement('input');input.value=modelSettings.values[role];input.placeholder=modelSettings.models[role].filename;input.setAttribute('aria-label',t('{model} path',{model:t(key)}));pathInputs[role]=input;
      const button=document.createElement('button');button.type='button';button.dataset.i18n='Choose…';button.textContent=t('Choose…');button.onclick=()=>pick(role);row.append(input,button);
      const note=document.createElement('div');note.className='model-status';note.dataset.i18n=modelSettings.models[role].exists?'File found · Compatibility checked at engine startup':'No model selected';note.textContent=t(note.dataset.i18n);box.append(label,row,note);
    }
    $('download-dir').value=modelSettings.values.download_dir;
    $('api-example').textContent=`curl ${location.origin}/api/engine/start -H 'Content-Type: application/json' -d '{}'\n\ncurl ${location.origin}/api/generate -H 'Content-Type: application/json' -d '{"prompt":"a red teapot","seed":"42","mode":"resolution","width":512,"height":512,"steps":8,"step_cache":false}'\n\ncurl ${location.origin}/api/state`;
    await downloadRefresh();updateDownloadFields();
  }catch(error){message(error.message);}
}
$('settings-open').onclick=()=>{$('settings-panel').hidden=false;loadSettings();if(!downloadTimer)downloadTimer=setInterval(()=>{if(!$('settings-panel').hidden)downloadRefresh();},1500);};
$('settings-close').onclick=()=>{$('settings-panel').hidden=true;};
$('settings-save').onclick=async()=>{try{const data={download_dir:$('download-dir').value};for(const role of Object.keys(pathInputs))data[role]=pathInputs[role].value;await post('/api/settings',data);message(t('Settings saved. Start inference to load the models.'));await loadSettings();}catch(error){message(error.message);}};
function updateDownloadFields(){
  $('custom-download').hidden=$('download-source').value==='huggingface';
  if(downloadCatalog){$('download-sha').value=downloadCatalog.models[$('download-role').value].sha256;$('model-license').href=downloadCatalog.license_url;}
}
$('download-source').onchange=updateDownloadFields;$('download-role').onchange=updateDownloadFields;
async function downloadRefresh(){
  try{const state=await jsonGet('/api/downloads');downloadCatalog=state.catalog;downloadTask=state.task;
    const task=state.task,running=task&&['downloading','verifying'].includes(task.status);
    $('download-start').disabled=!!running;$('download-cancel').disabled=!running;$('download-use').hidden=!task||task.status!=='completed';$('download-progress').hidden=!running||!task.total;
    if(task?.total)$('download-progress').value=task.received/task.total*100;
    $('download-status').textContent=task?`${t(({downloading:'Downloading',verifying:'Verifying SHA-256',completed:'Download complete',cancelled:'Cancelled; resume available',failed:'Download failed'})[task.status]||task.status)} · ${(task.received/1e9).toFixed(2)} / ${task.total?(task.total/1e9).toFixed(2):'?'} GB${task.error?' · '+task.error:''}`:t('No download yet');
  }catch(error){message(error.message);}
}
$('download-start').onclick=async()=>{try{await post('/api/downloads/start',{role:$('download-role').value,source:$('download-source').value==='huggingface'?'huggingface':'custom',url:$('download-url').value,sha256:$('download-sha').value});await downloadRefresh();}catch(error){message(error.message);}};
$('download-cancel').onclick=async()=>{try{await post('/api/downloads/cancel',{});await downloadRefresh();}catch(error){message(error.message);}};
$('download-use').onclick=()=>{if(downloadTask?.status==='completed'){pathInputs[downloadTask.role].value=downloadTask.path;message(t('Path filled in. Stop inference, then save the model paths.'));}};
function confirmDelete(id){if(deleteBusy)return;deleteTarget=id;$('delete-dialog').showModal();}
$('delete-result').onclick=()=>{if(selected)confirmDelete(selected);};
$('delete-dismiss').onclick=()=>$('delete-dialog').close();
$('delete-dialog').addEventListener('cancel',event=>{if(deleteBusy)event.preventDefault();});
$('delete-dialog').addEventListener('close',()=>{if(!deleteBusy)deleteTarget=null;});
$('delete-confirm').onclick=async()=>{
  if(!deleteTarget||deleteBusy)return;
  deleteBusy=true;$('delete-confirm').disabled=$('delete-dismiss').disabled=true;
  try{await post('/api/jobs/'+encodeURIComponent(deleteTarget)+'/delete',{});if(selected===deleteTarget)selected=null;gallerySignature='';$('delete-dialog').close();await refresh();
    if(!lastState?.jobs.some(job=>job.status==='completed')){$('result-image').hidden=true;$('empty').hidden=false;$('download').hidden=true;$('delete-result').hidden=true;$('status').textContent=t('Image moved to Trash');$('result-info').textContent=t('MLX / Metal · Local inference');$('logs').textContent=t('Waiting for a generation job…');}
  }catch(error){$('delete-dialog').close();showError(error.message);}
  finally{deleteBusy=false;deleteTarget=null;$('delete-confirm').disabled=$('delete-dismiss').disabled=false;gallerySignature='';await refresh();}
};
window.addEventListener('mqwen-language-change',()=>{
  showError('');message('');gallerySignature='';updateSize();updateSteps();
  // Do not reload settings or touch editable values while switching language.
  for(const [role,key] of Object.entries({dit:'Turbo DiT',text_encoder:'Text encoder',vae:'VAE'}))if(pathInputs[role])pathInputs[role].setAttribute('aria-label',t('{model} path',{model:t(key)}));
  if(!selected)$('status').textContent=t('Ready');
  refresh();if(!$('settings-panel').hidden)downloadRefresh();
});
setMode('resolution');updateSteps();refresh();setInterval(refresh,1500);
})();
