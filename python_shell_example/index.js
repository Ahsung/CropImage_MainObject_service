var fs = require("fs");
var pyshell = require(__dirname + "/user_modules/pyshell");

var data = fs.readFileSync("inputData");
var dataBase64 = data.toString("base64");

console.log("send!");

// script,python,callbackFunction
pyshell.runbase64(
  dataBase64,
  "python_Script_path.py",
  "", //empty str is default python,  other ex) "python3.6" or "/home/user/tensorflowenv/bin/python3" etc...
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
