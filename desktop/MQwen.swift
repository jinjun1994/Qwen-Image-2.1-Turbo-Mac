import Cocoa
import WebKit
import UniformTypeIdentifiers

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler, WKDownloadDelegate {
    var window: NSWindow!
    var web: WKWebView!
    var worker: Process?
    var timer: Timer?
    var status: NSTextField!
    var portField: NSTextField!
    var portLabel: NSTextField!
    var startButton: NSButton!
    var stopButton: NSButton!
    var browserButton: NSButton!
    var outputButton: NSButton!
    var logsButton: NSButton!
    var languagePicker: NSPopUpButton!
    var quitting = false
    var stopping = false
    var language = UserDefaults.standard.string(forKey: "interfaceLanguage") == "zh" ? "zh" : "en"
    var dictionary = [String:String]()
    var statusKey = "Service stopped"
    var statusValues = [String:String]()
    let token = UUID().uuidString
    var port = UserDefaults.standard.integer(forKey: "serverPort")
    let support = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/MQwen")
    var base: URL { URL(string: "http://127.0.0.1:\(port)")! }
    var version: String { Bundle.main.object(forInfoDictionaryKey:"CFBundleShortVersionString") as? String ?? "0.4.0" }
    func tr(_ key:String,_ values:[String:String]=[:])->String {
        var result=language == "zh" ? (dictionary[key] ?? key) : key
        for (name,value) in values {result=result.replacingOccurrences(of:"{\(name)}",with:value)}
        return result
    }
    func setStatus(_ key:String,_ values:[String:String]=[:]) {statusKey=key;statusValues=values;status.stringValue=tr(key,values)}
    func menus() {
        let menu=NSMenu(), appMenu=NSMenu()
        let aboutItem=appMenu.addItem(withTitle:tr("About MQwen"),action:#selector(about),keyEquivalent:"");aboutItem.target=self
        appMenu.addItem(NSMenuItem.separator())
        appMenu.addItem(withTitle:tr("Quit MQwen"),action:#selector(NSApplication.terminate(_:)),keyEquivalent:"q")
        let appItem=NSMenuItem();appItem.submenu=appMenu;menu.addItem(appItem)
        let edit=NSMenuItem();let editMenu=NSMenu(title:tr("Edit"))
        for (title,action,key) in [("Cut","cut:","x"),("Copy","copy:","c"),("Paste","paste:","v"),("Select All","selectAll:","a")] {editMenu.addItem(withTitle:tr(title),action:Selector(action),keyEquivalent:key)}
        edit.submenu=editMenu;menu.addItem(edit);NSApp.mainMenu=menu
    }
    func applyLanguage() {
        menus();window.title=tr("MQwen · Local Image Studio")
        portLabel.stringValue=tr("Port")
        for (button,key) in [(startButton,"Start WebUI / API"),(stopButton,"Stop WebUI / API"),(browserButton,"Open in browser"),(outputButton,"Images folder"),(logsButton,"Service logs")] {button?.title=tr(key)}
        languagePicker.selectItem(at:language == "zh" ? 1:0)
        status.stringValue=tr(statusKey,statusValues)
    }
    @objc func changeLanguage() {setLanguage(languagePicker.indexOfSelectedItem == 1 ? "zh":"en")}
    func setLanguage(_ next:String) {
        guard ["en","zh"].contains(next) else{return}
        language=next;UserDefaults.standard.set(next,forKey:"interfaceLanguage");applyLanguage()
        if worker == nil {showStopped()}
        else {web.evaluateJavaScript("window.mqwenSetLanguage && window.mqwenSetLanguage('\(next)', false)",completionHandler:nil)}
    }
    func applicationDidFinishLaunching(_ notification: Notification) {
        if let resources=Bundle.main.resourceURL,
           let data=try? Data(contentsOf:resources.appendingPathComponent("worker/_internal/mqwen/assets/web/i18n.json")),
           let values=try? JSONSerialization.jsonObject(with:data) as? [String:String] {dictionary=values}
        if port < 1024 {port=7861}
        window=NSWindow(contentRect:NSRect(x:0,y:0,width:1280,height:900),styleMask:[.titled,.closable,.miniaturizable,.resizable],backing:.buffered,defer:false)
        window.minSize=NSSize(width:1050,height:700);window.center()
        let root=NSView();window.contentView=root
        let bar=NSStackView();bar.orientation = .horizontal;bar.spacing=8;bar.translatesAutoresizingMaskIntoConstraints=false
        startButton=NSButton(title:"",target:self,action:#selector(startServer));stopButton=NSButton(title:"",target:self,action:#selector(stopServer))
        browserButton=NSButton(title:"",target:self,action:#selector(openBrowser));outputButton=NSButton(title:"",target:self,action:#selector(openOutputs));logsButton=NSButton(title:"",target:self,action:#selector(openLogs))
        portLabel=NSTextField(labelWithString:"");portField=NSTextField(string:String(port));portField.widthAnchor.constraint(equalToConstant:62).isActive=true
        status=NSTextField(labelWithString:"");status.setContentCompressionResistancePriority(.defaultLow,for:.horizontal)
        languagePicker=NSPopUpButton(frame:.zero,pullsDown:false);languagePicker.addItems(withTitles:["English","中文"]);languagePicker.target=self;languagePicker.action=#selector(changeLanguage);languagePicker.setAccessibilityLabel("Language / 语言")
        for view in [portLabel!,portField!,startButton!,stopButton!,browserButton!,outputButton!,logsButton!,languagePicker!,status!] as [NSView] {bar.addArrangedSubview(view)}
        let config=WKWebViewConfiguration();config.userContentController.add(self,name:"mqwen")
        web=WKWebView(frame:.zero,configuration:config);web.navigationDelegate=self;web.uiDelegate=self;web.translatesAutoresizingMaskIntoConstraints=false
        root.addSubview(bar);root.addSubview(web)
        NSLayoutConstraint.activate([bar.topAnchor.constraint(equalTo:root.topAnchor,constant:10),bar.leadingAnchor.constraint(equalTo:root.leadingAnchor,constant:14),bar.trailingAnchor.constraint(lessThanOrEqualTo:root.trailingAnchor,constant:-14),bar.heightAnchor.constraint(equalToConstant:34),web.topAnchor.constraint(equalTo:bar.bottomAnchor,constant:10),web.leadingAnchor.constraint(equalTo:root.leadingAnchor),web.trailingAnchor.constraint(equalTo:root.trailingAnchor),web.bottomAnchor.constraint(equalTo:root.bottomAnchor)])
        applyLanguage();window.makeKeyAndOrderFront(nil);NSApp.activate(ignoringOtherApps:true);showStopped();startServer()
    }
    @objc func about() {
        let alert=NSAlert();alert.messageText="MQwen \(version)";alert.informativeText=tr("Qwen-Image 2.1 Turbo · MLX / Metal\nLocal image generation on Apple Silicon\nSource: MIT. See README and THIRD_PARTY for dependency and model licenses.");alert.addButton(withTitle:tr("OK"));alert.runModal()
    }
    func showStopped() {
        web.loadHTMLString("<html lang='\(language)'><body style='background:#111310;font:20px -apple-system;padding:90px;color:#eff0e8'><h1>MQwen</h1><p>\(tr("Local Image Studio"))</p><p>\(tr("Click Start WebUI / API above to begin. Models load only when you start inference."))</p></body></html>",baseURL:nil)
        stopButton.isEnabled=false;startButton.isEnabled=true;portField.isEnabled=true
    }
    @objc func startServer() {
        guard worker == nil else{return}
        guard let value=Int(portField.stringValue),(1024...65535).contains(value) else {setStatus("Port must be between 1024 and 65535");return}
        port=value;UserDefaults.standard.set(port,forKey:"serverPort")
        // Never attach to or stop an unrelated service at this port.
        var address=sockaddr_in();address.sin_family=sa_family_t(AF_INET);address.sin_port=UInt16(port).bigEndian;address.sin_addr.s_addr=inet_addr("127.0.0.1")
        let fd=socket(AF_INET,SOCK_STREAM,0);var reuse:Int32=1
        setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&reuse,socklen_t(MemoryLayout<Int32>.size))
        let available=withUnsafePointer(to:&address){ptr in ptr.withMemoryRebound(to:sockaddr.self,capacity:1){Darwin.bind(fd,$0,socklen_t(MemoryLayout<sockaddr_in>.size))==0}}
        close(fd);guard available else {setStatus("Port is in use; choose another");return}
        do {
            try FileManager.default.createDirectory(at:support,withIntermediateDirectories:true)
            let output=FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Pictures/MQwen")
            try FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
            let logURL=support.appendingPathComponent("server.log")
            if !FileManager.default.fileExists(atPath:logURL.path) {FileManager.default.createFile(atPath:logURL.path,contents:nil)}
            let log=try FileHandle(forWritingTo:logURL);try log.seekToEnd()
            let process=Process();process.executableURL=Bundle.main.resourceURL!.appendingPathComponent("worker/mqwen-worker")
            process.arguments=["web","--desktop","--no-auto-start","--port",String(port),"--output-dir",output.path,"--config-dir",support.path]
            var env=ProcessInfo.processInfo.environment;env["MQWEN_CONTROL_TOKEN"]=token;env["PYTHONUNBUFFERED"]="1";env["USE_TORCH"]="0";env["USE_TF"]="0"
            process.environment=env;process.currentDirectoryURL=support;process.standardOutput=log;process.standardError=log
            process.terminationHandler={ [weak self] p in DispatchQueue.main.async {guard let self=self else{return};self.timer?.invalidate();self.worker=nil;self.stopping=false;self.setStatus("Service stopped ({code})",["code":String(p.terminationStatus)]);self.showStopped();if self.quitting {NSApp.reply(toApplicationShouldTerminate:true)}}}
            try process.run();worker=process;startButton.isEnabled=false;stopButton.isEnabled=true;portField.isEnabled=false;setStatus("Starting service…")
            var attempts=0
            timer=Timer.scheduledTimer(withTimeInterval:0.5,repeats:true){[weak self] timer in
                guard let self=self else{return};attempts+=1
                URLSession.shared.dataTask(with:self.base.appendingPathComponent("api/health")){data,_,_ in
                    guard let data=data,let info=(try? JSONSerialization.jsonObject(with:data)) as? [String:Any],info["service"] as? String == "mqwen",info["pid"] as? Int == Int(process.processIdentifier) else{return}
                    DispatchQueue.main.async {guard self.worker===process,!self.stopping else{return};timer.invalidate();self.setStatus("API 127.0.0.1:{port}",["port":String(self.port)]);self.web.load(URLRequest(url:URL(string:self.base.absoluteString+"/?lang="+self.language)!))}
                }.resume()
                if attempts>=120 {timer.invalidate();self.setStatus("Startup timed out; check the logs")}
            }
        } catch {setStatus(error.localizedDescription);worker=nil;showStopped()}
    }
    @objc func stopServer() {
        guard let process=worker,!stopping else{return};stopping=true;timer?.invalidate();stopButton.isEnabled=false;setStatus("Waiting for GPU operations and releasing models…")
        var req=URLRequest(url:base.appendingPathComponent("api/server/shutdown"));req.httpMethod="POST";req.setValue("application/json",forHTTPHeaderField:"Content-Type");req.setValue(token,forHTTPHeaderField:"X-MQwen-Control");req.httpBody=Data("{}".utf8)
        URLSession.shared.dataTask(with:req){_,response,_ in if (response as? HTTPURLResponse)?.statusCode != 202,process.isRunning {process.terminate()}}.resume()
    }
    @objc func openBrowser(){if worker != nil && !stopping {NSWorkspace.shared.open(URL(string:base.absoluteString+"/?lang="+language)!)}}
    @objc func openOutputs(){NSWorkspace.shared.open(FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Pictures/MQwen"))}
    @objc func openLogs(){NSWorkspace.shared.open(support.appendingPathComponent("server.log"))}
    func applicationShouldTerminateAfterLastWindowClosed(_ sender:NSApplication)->Bool {true}
    func applicationShouldTerminate(_ sender:NSApplication)->NSApplication.TerminateReply {if worker==nil{return .terminateNow};quitting=true;stopServer();return .terminateLater}
    func webView(_ webView:WKWebView,didFinish navigation:WKNavigation!) {
        guard webView.url?.host=="127.0.0.1" else{return}
        webView.evaluateJavaScript("window.mqwenLocaleReady && window.mqwenLocaleReady.then(() => window.mqwenSetLanguage('\(language)', false))",completionHandler:nil)
    }
    func webView(_ webView:WKWebView,decidePolicyFor action:WKNavigationAction,decisionHandler:@escaping(WKNavigationActionPolicy)->Void) {
        guard let url=action.request.url else {decisionHandler(.cancel);return}
        if action.shouldPerformDownload {decisionHandler(.download);return}
        if url.scheme=="about" || (url.scheme=="http" && url.host=="127.0.0.1" && url.port==port) {decisionHandler(.allow)}
        else {if ["https","http"].contains(url.scheme ?? ""){NSWorkspace.shared.open(url)};decisionHandler(.cancel)}
    }
    func webView(_ webView:WKWebView,navigationAction:WKNavigationAction,didBecome download:WKDownload){download.delegate=self}
    func download(_ download:WKDownload,decideDestinationUsing response:URLResponse,suggestedFilename:String,completionHandler:@escaping(URL?)->Void) {let panel=NSSavePanel();panel.title=tr("Save PNG");panel.prompt=tr("Save");panel.nameFieldStringValue=suggestedFilename;panel.beginSheetModal(for:window){completionHandler($0 == .OK ? panel.url:nil)}}
    func userContentController(_ controller:WKUserContentController,didReceive message:WKScriptMessage) {
        guard message.frameInfo.isMainFrame,message.frameInfo.securityOrigin.protocol=="http",message.frameInfo.securityOrigin.host=="127.0.0.1",message.frameInfo.securityOrigin.port==port,let body=message.body as? [String:String] else{return}
        if body["action"]=="language",let next=body["language"] {setLanguage(next);return}
        guard let key=body["key"],["dit","text_encoder","vae","download_dir"].contains(key) else{return}
        let panel=NSOpenPanel();panel.canChooseDirectories=key=="download_dir";panel.canChooseFiles=key != "download_dir";panel.allowsMultipleSelection=false;panel.title=tr(key=="download_dir" ? "Choose a download folder":"Choose a model file");panel.prompt=tr("Choose")
        panel.beginSheetModal(for:window){result in
            guard result == .OK,let path=panel.url?.path,let data=try? JSONSerialization.data(withJSONObject:["key":key,"path":path]),let json=String(data:data,encoding:.utf8) else{return}
            self.web.evaluateJavaScript("window.mqwenPicked(\(json))",completionHandler:nil)
        }
    }
}
let app=NSApplication.shared
let delegate=AppDelegate();app.delegate=delegate;app.setActivationPolicy(.regular);app.run()
