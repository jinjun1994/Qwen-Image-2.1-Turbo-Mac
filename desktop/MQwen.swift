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
    var startButton: NSButton!
    var stopButton: NSButton!
    var quitting = false
    var stopping = false
    let token = UUID().uuidString
    var port = UserDefaults.standard.integer(forKey: "serverPort")
    let support = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/MQwen")
    var base: URL { URL(string: "http://127.0.0.1:\(port)")! }

    func applicationDidFinishLaunching(_ notification: Notification) {
        if port < 1024 { port = 7861 }
        let menu = NSMenu()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "关于 MQwen", action: #selector(about), keyEquivalent: "")
        appMenu.addItem(NSMenuItem.separator())
        appMenu.addItem(withTitle: "退出 MQwen", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let appItem = NSMenuItem(); appItem.submenu = appMenu; menu.addItem(appItem)
        let edit = NSMenuItem(); let editMenu = NSMenu(title: "编辑")
        for (title, action, key) in [("剪切", "cut:", "x"), ("复制", "copy:", "c"), ("粘贴", "paste:", "v"), ("全选", "selectAll:", "a")] { editMenu.addItem(withTitle: title, action: Selector(action), keyEquivalent: key) }
        edit.submenu=editMenu; menu.addItem(edit); NSApp.mainMenu=menu
        window=NSWindow(contentRect:NSRect(x:0,y:0,width:1280,height:900), styleMask:[.titled,.closable,.miniaturizable,.resizable], backing:.buffered,defer:false)
        window.title="MQwen · 本地影像工作室"; window.minSize=NSSize(width:960,height:700);window.center()
        let root=NSView(); window.contentView=root
        let bar=NSStackView();bar.orientation = .horizontal;bar.spacing=10;bar.translatesAutoresizingMaskIntoConstraints=false
        startButton=NSButton(title:"启动 WebUI / API",target:self,action:#selector(startServer))
        stopButton=NSButton(title:"关闭 WebUI / API",target:self,action:#selector(stopServer))
        let browser=NSButton(title:"在浏览器打开",target:self,action:#selector(openBrowser))
        let output=NSButton(title:"作品文件夹",target:self,action:#selector(openOutputs))
        let logs=NSButton(title:"服务日志",target:self,action:#selector(openLogs))
        portField=NSTextField(string:String(port));portField.widthAnchor.constraint(equalToConstant:65).isActive=true
        status=NSTextField(labelWithString:"服务未启动");status.setContentCompressionResistancePriority(.defaultLow,for:.horizontal)
        for view in [NSTextField(labelWithString:"端口"),portField!,startButton!,stopButton!,browser,output,logs,status!] {bar.addArrangedSubview(view)}
        let config=WKWebViewConfiguration();config.userContentController.add(self,name:"mqwen")
        web=WKWebView(frame:.zero,configuration:config);web.navigationDelegate=self;web.uiDelegate=self;web.translatesAutoresizingMaskIntoConstraints=false
        root.addSubview(bar);root.addSubview(web)
        NSLayoutConstraint.activate([bar.topAnchor.constraint(equalTo:root.topAnchor,constant:10),bar.leadingAnchor.constraint(equalTo:root.leadingAnchor,constant:14),bar.trailingAnchor.constraint(lessThanOrEqualTo:root.trailingAnchor,constant:-14),bar.heightAnchor.constraint(equalToConstant:34),web.topAnchor.constraint(equalTo:bar.bottomAnchor,constant:10),web.leadingAnchor.constraint(equalTo:root.leadingAnchor),web.trailingAnchor.constraint(equalTo:root.trailingAnchor),web.bottomAnchor.constraint(equalTo:root.bottomAnchor)])
        window.makeKeyAndOrderFront(nil);NSApp.activate(ignoringOtherApps:true)
        showStopped();startServer()
    }
    @objc func about() {let alert=NSAlert();alert.messageText="MQwen 0.3.0";alert.informativeText="Qwen-Image 2.1 Turbo · MLX / Metal\nApple Silicon 本地生图\n代码 MIT；依赖及模型许可见随附 README 与 THIRD_PARTY。";alert.runModal()}
    func showStopped() {web.loadHTMLString("<html><body style='background:#f5f3ee;font:20px -apple-system;padding:90px;color:#262622'><h1>MQwen</h1><p>本地影像工作室</p><p>点击上方「启动 WebUI / API」开始。模型会在你点击启动推理服务后加载。</p></body></html>",baseURL:nil);stopButton.isEnabled=false;startButton.isEnabled=true;portField.isEnabled=true}
    @objc func startServer() {
        guard worker == nil else {return}
        guard let value=Int(portField.stringValue), (1024...65535).contains(value) else {status.stringValue="端口须为 1024–65535";return}
        port=value;UserDefaults.standard.set(port,forKey:"serverPort")
        // Do not attach to or terminate an unrelated service at the chosen port.
        var address=sockaddr_in();address.sin_family=sa_family_t(AF_INET);address.sin_port=UInt16(port).bigEndian;address.sin_addr.s_addr=inet_addr("127.0.0.1")
        let fd=socket(AF_INET,SOCK_STREAM,0)
        var reuse:Int32=1
        setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&reuse,socklen_t(MemoryLayout<Int32>.size))
        let available=withUnsafePointer(to:&address){ptr in ptr.withMemoryRebound(to:sockaddr.self,capacity:1){Darwin.bind(fd,$0,socklen_t(MemoryLayout<sockaddr_in>.size))==0}}
        close(fd)
        guard available else {status.stringValue="端口已被占用，请更换";return}
        do {
            try FileManager.default.createDirectory(at:support,withIntermediateDirectories:true)
            let output=FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Pictures/MQwen")
            try FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
            let logURL=support.appendingPathComponent("server.log")
            if !FileManager.default.fileExists(atPath:logURL.path) {FileManager.default.createFile(atPath:logURL.path,contents:nil)}
            let log=try FileHandle(forWritingTo:logURL);try log.seekToEnd()
            let process=Process()
            process.executableURL=Bundle.main.resourceURL!.appendingPathComponent("worker/mqwen-worker")
            process.arguments=["web","--desktop","--no-auto-start","--port",String(port),"--output-dir",output.path,"--config-dir",support.path]
            var env=ProcessInfo.processInfo.environment
            env["MQWEN_CONTROL_TOKEN"]=token;env["PYTHONUNBUFFERED"]="1";env["USE_TORCH"]="0";env["USE_TF"]="0"
            process.environment=env;process.currentDirectoryURL=support;process.standardOutput=log;process.standardError=log
            process.terminationHandler={ [weak self] p in DispatchQueue.main.async {guard let self=self else{return};self.timer?.invalidate();self.worker=nil;self.stopping=false;self.status.stringValue="服务已停止 (\(p.terminationStatus))";self.showStopped();if self.quitting {NSApp.reply(toApplicationShouldTerminate:true)}}}
            try process.run();worker=process;startButton.isEnabled=false;stopButton.isEnabled=true;portField.isEnabled=false;status.stringValue="服务启动中…"
            var attempts=0
            timer=Timer.scheduledTimer(withTimeInterval:0.5,repeats:true){[weak self] timer in
                guard let self=self else{return};attempts+=1
                let session=URLSession(configuration:.ephemeral)
                session.dataTask(with:self.base.appendingPathComponent("api/health")){data,_,_ in
                    guard let data=data,let info=(try? JSONSerialization.jsonObject(with:data)) as? [String:Any],info["service"] as? String == "mqwen",info["pid"] as? Int == Int(process.processIdentifier) else {return}
                    DispatchQueue.main.async {guard self.worker===process,!self.stopping else{return};timer.invalidate();self.status.stringValue="API 127.0.0.1:\(self.port)";self.web.load(URLRequest(url:self.base))}
                }.resume()
                if attempts>=120 {timer.invalidate();self.status.stringValue="启动超时，请查看日志"}
            }
        } catch {status.stringValue=error.localizedDescription;worker=nil;showStopped()}
    }
    @objc func stopServer() {
        guard let process=worker,!stopping else{return};stopping=true;timer?.invalidate();stopButton.isEnabled=false;status.stringValue="等待当前 GPU 运算结束并释放模型…"
        var req=URLRequest(url:base.appendingPathComponent("api/server/shutdown"));req.httpMethod="POST";req.setValue("application/json",forHTTPHeaderField:"Content-Type");req.setValue(token,forHTTPHeaderField:"X-MQwen-Control");req.httpBody=Data("{}".utf8)
        URLSession.shared.dataTask(with:req){_,response,_ in
            if (response as? HTTPURLResponse)?.statusCode != 202 {if process.isRunning {process.terminate()}}
        }.resume()
    }
    @objc func openBrowser(){if worker != nil && !stopping {NSWorkspace.shared.open(base)}}
    @objc func openOutputs(){NSWorkspace.shared.open(FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Pictures/MQwen"))}
    @objc func openLogs(){NSWorkspace.shared.open(support.appendingPathComponent("server.log"))}
    func applicationShouldTerminateAfterLastWindowClosed(_ sender:NSApplication)->Bool {true}
    func applicationShouldTerminate(_ sender:NSApplication)->NSApplication.TerminateReply {if worker==nil{return .terminateNow};quitting=true;stopServer();return .terminateLater}
    func webView(_ webView:WKWebView,decidePolicyFor action:WKNavigationAction,decisionHandler:@escaping(WKNavigationActionPolicy)->Void) {
        guard let url=action.request.url else {decisionHandler(.cancel);return}
        if action.shouldPerformDownload {decisionHandler(.download);return}
        if url.scheme=="about" || (url.scheme=="http" && url.host=="127.0.0.1" && url.port==port) {decisionHandler(.allow)}
        else {if ["https","http"].contains(url.scheme ?? ""){NSWorkspace.shared.open(url)};decisionHandler(.cancel)}
    }
    func webView(_ webView:WKWebView,navigationAction:WKNavigationAction,didBecome download:WKDownload){download.delegate=self}
    func download(_ download:WKDownload,decideDestinationUsing response:URLResponse,suggestedFilename:String,completionHandler:@escaping(URL?)->Void) {let panel=NSSavePanel();panel.nameFieldStringValue=suggestedFilename;panel.beginSheetModal(for:window){completionHandler($0 == .OK ? panel.url:nil)}}
    func userContentController(_ controller:WKUserContentController,didReceive message:WKScriptMessage) {
        guard message.frameInfo.isMainFrame,message.frameInfo.securityOrigin.host=="127.0.0.1",message.frameInfo.securityOrigin.port==port,let body=message.body as? [String:String],let key=body["key"],["dit","text_encoder","vae","download_dir"].contains(key) else{return}
        let panel=NSOpenPanel();panel.canChooseDirectories=key=="download_dir";panel.canChooseFiles=key != "download_dir";panel.allowsMultipleSelection=false
        panel.beginSheetModal(for:window){result in
            guard result == .OK,let path=panel.url?.path,let data=try? JSONSerialization.data(withJSONObject:["key":key,"path":path]),let json=String(data:data,encoding:.utf8) else{return}
            self.web.evaluateJavaScript("window.mqwenPicked(\(json))",completionHandler:nil)
        }
    }
}
let app=NSApplication.shared
let delegate=AppDelegate();app.delegate=delegate;app.setActivationPolicy(.regular);app.run()
