/* Run with: node --test tests/test_i18n_browser.cjs (no browser packages needed). */
const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const source=fs.readFileSync(path.join(root,'src/mqwen/assets/web/i18n.js'),'utf8');
const dictionary=JSON.parse(fs.readFileSync(path.join(root,'src/mqwen/assets/web/i18n.json'),'utf8'));
async function boot(stored,query='',blockedStorage=false){
 const store=new Map(stored?[['mqwen.language',stored]]:[]),events=[],messages=[];
 const label={dataset:{i18n:'Start inference'},textContent:''};
 const prompt={value:'My own prompt 中文',placeholder:'',getAttribute:()=> 'Example: A red ceramic teapot on a wooden table, soft morning light, realistic photograph.',setAttribute(name,value){this[name]=value;}};
 const select={value:'',onchange:null};
 const document={documentElement:{lang:''},getElementById:()=>select,querySelectorAll(selector){if(selector==='[data-i18n]')return [label];if(selector==='[data-i18n-placeholder]')return [prompt];return [];}};
 const localStorage={getItem(key){if(blockedStorage)throw Error('Storage unavailable');return store.get(key)||null;},setItem(key,value){if(blockedStorage)throw Error('Storage unavailable');store.set(key,value);}};
 const window={dispatchEvent:e=>events.push(e.type),webkit:{messageHandlers:{mqwen:{postMessage:m=>messages.push(m)}}}};
 const location={search:query,href:'http://127.0.0.1:7861/'+query};
 const history={replaceState(_state,_title,url){location.search=url.search;location.href=url.href;}};
 vm.runInNewContext(source,{window,document,localStorage,location,history,URL,URLSearchParams,CustomEvent:class{constructor(type){this.type=type;}},console,fetch:async()=>({ok:true,json:async()=>dictionary})});
 await window.mqwenLocaleReady;
 return {window,document,label,prompt,select,store,messages,events};
}
test('English default; language changes preserve inputs and synchronize native once',async()=>{
 const app=await boot();assert.equal(app.document.documentElement.lang,'en');assert.equal(app.label.textContent,'Start inference');
 app.window.mqwenSetLanguage('zh');assert.equal(app.label.textContent,'启动推理服务');assert.equal(app.document.documentElement.lang,'zh-CN');assert.equal(app.store.get('mqwen.language'),'zh');assert.equal(app.messages.length,1);assert.equal(app.prompt.value,'My own prompt 中文');
 app.window.mqwenSetLanguage('en',false);assert.equal(app.label.textContent,'Start inference');assert.equal(app.messages.length,1);assert.equal(app.prompt.value,'My own prompt 中文');
});
test('Stored preference, explicit desktop locale, invalid locale and storage failures',async()=>{
 assert.equal((await boot('zh')).label.textContent,'启动推理服务');
 assert.equal((await boot('zh','?lang=en')).label.textContent,'Start inference');
 const app=await boot('invalid');assert.equal(app.label.textContent,'Start inference');app.window.mqwenSetLanguage('fr');assert.equal(app.window.mqwenLocale.language,'en');
 const limited=await boot(null,'',true);limited.window.mqwenSetLanguage('zh');assert.equal(limited.label.textContent,'启动推理服务');
});
