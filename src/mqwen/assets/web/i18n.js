/* Shared browser/desktop locale. English is the default, regardless of OS locale. */
window.mqwenLocaleReady=(async()=>{
  let stored='en';
  try{stored=localStorage.getItem('mqwen.language')||'en';}catch{}
  const query=new URLSearchParams(location.search).get('lang');
  let language=['en','zh'].includes(query)?query:(['en','zh'].includes(stored)?stored:'en');
  let dictionary={};
  try{const response=await fetch('/i18n.json');if(!response.ok)throw Error('Translations unavailable');dictionary=await response.json();}
  catch(error){console.warn(error);}
  const t=(key,values={})=>{
    const template=language==='zh'?(dictionary[key]||key):key;
    return template.replace(/\{(\w+)\}/g,(match,name)=>Object.hasOwn(values,name)?String(values[name]):match);
  };
  function apply(){
    document.documentElement.lang=language==='zh'?'zh-CN':'en';
    document.querySelectorAll('[data-i18n]').forEach(el=>{el.textContent=t(el.dataset.i18n);});
    for(const attr of ['placeholder','title','aria-label','alt']){
      document.querySelectorAll(`[data-i18n-${attr}]`).forEach(el=>el.setAttribute(attr,t(el.getAttribute(`data-i18n-${attr}`))));
    }
    document.getElementById('language').value=language;
  }
  window.mqwenLocale={t,get language(){return language;}};
  window.mqwenSetLanguage=(next,notifyNative=true)=>{
    if(!['en','zh'].includes(next))return;
    language=next;
    try{localStorage.setItem('mqwen.language',next);}catch{}
    const url=new URL(location.href);url.searchParams.set('lang',next);history.replaceState(null,'',url);
    apply();
    window.dispatchEvent(new CustomEvent('mqwen-language-change'));
    if(notifyNative)window.webkit?.messageHandlers?.mqwen?.postMessage({action:'language',language:next});
  };
  document.getElementById('language').onchange=event=>window.mqwenSetLanguage(event.target.value);
  // Persist an explicit desktop/query locale as well as a user selection.
  try{localStorage.setItem('mqwen.language',language);}catch{}
  apply();
})();
