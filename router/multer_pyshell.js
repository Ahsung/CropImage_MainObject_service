var pyshell = require("../user_modules/pyshell");

module.exports = function(app, upload) {
  app.get("/", function(req, res) {
    res.render("b_multer.html");
  });

  app.post("/", upload, function(req, res) {
    console.log("uplode complete");
    if (!req.file) {
      console.log("file 제출 안함.");
      res.end("No File");
      return;
    }
    var dataBase64 = req.file.buffer.toString("base64");

    pyshell.runbase64(
      dataBase64,
      "/home/asung/detectron2/test/web_base64_rate.py",
      "/home/asung/detectron2/detectron2/bin/python3",

      function(err, result) {
        if (err) {
          console.log(err);
          res.end("not image_file");
        } else {
          //encode bytes
          imdata = Buffer.from(result, "base64");
          console.log("python complete");
          console.log(imdata);

          //image send
          res.writeHead(200, {
            "Content-Type": "image/jpeg",
            "Content-Length": imdata.length
          });
          res.end(imdata);
        }
      }
    );
  });
};
