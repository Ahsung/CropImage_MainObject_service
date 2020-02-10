var express = require("express");
var fs = require("fs");
var app = express();
var pyshell = require("/home/asung/nodejsProject/nodetst/user_modules/pyshell");

var data = fs.readFileSync("/home/asung/detImage/i3.jpeg");

var dataBase64 = data.toString("base64");

console.log("전송!");
dat = {
  binary: data
};

pyshell.runbase64(
  dataBase64,
  "/home/asung/detectron2/test/wd_base64.py",
  "/home/asung/detectron2/detectron2/bin/python3",
  function(err, result) {
    if (err) {
      console.log(err);
    } else {
      imdata = Buffer.from(result, "base64");
      console.log(imdata);
      fs.writeFileSync("result.jpeg", imdata);
      console.log("complete");
    }
  }
);
