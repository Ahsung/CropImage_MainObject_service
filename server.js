const express = require("express");
const app = express();

var multer = require("multer");
var upload = multer({ storage: multer.memoryStorage() }).single("userfile");

var router = require(__dirname + "/router/multer_pyshell")(app, upload);
//html path
app.set("views", __dirname + "/views");
//render ejs
app.set("view engine", "ejs");
app.engine("html", require("ejs").renderFile);

//linsen
const port = process.env.PORT || 9000;
app.listen(9000, function() {
  console.log(`port ${port} is runnung`);
});
