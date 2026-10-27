import Cocoa
let destination=CommandLine.arguments[1]
try FileManager.default.createDirectory(atPath:destination,withIntermediateDirectories:true)
for size in [16,32,64,128,256,512,1024] {
 let image=NSImage(size:NSSize(width:size,height:size));image.lockFocus()
 let scale=CGFloat(size)/1024
 NSAffineTransform(scale:scale).concat()
 NSColor(calibratedRed:0.08,green:0.10,blue:0.08,alpha:1).setFill()
 NSBezierPath(roundedRect:NSRect(x:0,y:0,width:1024,height:1024),xRadius:220,yRadius:220).fill()
 NSColor(calibratedRed:0.84,green:0.93,blue:0.61,alpha:1).setStroke()
 let ring=NSBezierPath(ovalIn:NSRect(x:225,y:240,width:565,height:565));ring.lineWidth=78;ring.stroke()
 let tail=NSBezierPath();tail.move(to:NSPoint(x:585,y:410));tail.line(to:NSPoint(x:795,y:190));tail.lineWidth=78;tail.lineCapStyle = .round;tail.stroke()
 image.unlockFocus()
 let bitmap=NSBitmapImageRep(data:image.tiffRepresentation!)!
 try bitmap.representation(using:.png,properties:[:])!.write(to:URL(fileURLWithPath:destination+"/\(size).png"))
}
extension NSAffineTransform {convenience init(scale:CGFloat){self.init();self.scale(by:scale)}}
