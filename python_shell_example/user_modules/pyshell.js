var Pythonshell = require("python-shell");

// send json
// callback return Json
function runJson(json, scriptPath, pythonPath, callback) {
  pyshell = new Pythonshell.PythonShell(scriptPath, {
    pythonPath: pythonPath
  });

  pyshell.send(JSON.stringify(json), { mode: "json" });

  pyshell.on("message", results => {
    var rejoson = JSON.parse(results).img;
    var err = null;
    callback(err, rejoson);
    return rejoson;
  });

  pyshell.end(err => {
    if (err) {
      var buffer = null;
      callback(err, buffer);
      return;
    }
  });
}

// send base64
// callback return base64
function runbase64(base64, scriptPath, pythonPath, callback) {
  pyshell = new Pythonshell.PythonShell(scriptPath, {
    pythonPath: pythonPath
  });

  pyshell.send(base64);

  pyshell.on("message", rebase64 => {
    var err = null;
    callback(err, rebase64);
    return rebase64;
  });

  pyshell.end(err => {
    if (err) {
      var buffer = null;
      callback(err, buffer);
      return;
    }
  });
}

module.exports.runJson = runJson;
module.exports.runbase64 = runbase64;
