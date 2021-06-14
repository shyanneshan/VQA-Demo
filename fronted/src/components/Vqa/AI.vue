<template>
  <el-container>
    <div class="top-bar">
      <div class="option-bar">
      </div>
      <el-main>
        <div>
          <el-container id="consult-div">
            <el-container id="chat-div">
              <el-main>
                <div class="option-block">
                  <template>
                    <el-select v-model="characterChosen" placeholder="Please"
                               @change="chooseCharacter">
                      <el-option
                          v-for="item in characterList"
                          :key="item.value"
                          :label="item.label"
                          :value="item.value">
                      </el-option>
                    </el-select>
                  </template>
                </div>
                <div id="chat-history-box">
                  <div id="chat-history" ref="chat"></div>
                </div>
              </el-main>
              <el-dialog
                  title="UPLOAD"
                  :visible.sync="uploadFormVisible"
                  width="640px"
                  append-to-body
                  :close-on-click-modal="false"
                  :close-on-press-escape="false"
                  :show-close="false"
              >
                <el-form
                    ref="uploadForm"
                    :model="uploadForm"
                    :rules="rules"
                    label-position="left"
                    label-width="130px"
                >
                  <el-form-item label="Question" prop="desc">
                    <el-input
                        type="textarea"
                        v-model="uploadForm.desc"
                    ></el-input>
                  </el-form-item>
                  <el-form-item label="UPDATE" prop="uploadFile">
                    <el-upload
                        ref="upload"
                        action=""
                        accept=".png, .jpg, .jpeg"
                        :multiple="false"
                        :file-list="fileList"
                        :show-file-list="true"
                        :http-request="uploadMedicalArchive"
                        :auto-upload="false"
                    >
                      <el-button slot="trigger" type="primary"
                      >BROWSE</el-button
                      >
                    </el-upload>
                  </el-form-item>
                </el-form>
                <div slot="footer">
                  <el-button type="primary" @click="upload">SUBMIT</el-button>
                  <el-button @click="cancelUpload">CANCEL</el-button>
                </div>
              </el-dialog>
              <el-footer>
                <el-container>
                  <div id="input">

                    <template>
                      <el-input
                          placeholder="please input through UPDATE button"
                          v-model="input"
                          @keyup.enter.native="send"
                          style="width: 90%; float: left"
                          :disabled="true"
                      >
                        <el-button slot="prepend" @click="openUploadDialog"
                        >UPDATE</el-button
                        >
                        <el-button slot="append" @click="send" style="float: left"
                        >SEND</el-button
                        >
                      </el-input>
                    </template>

                  </div>
                </el-container>
              </el-footer>
            </el-container>
          </el-container>
        </div>
      </el-main>
    </div>
  </el-container>
</template>

<script>
export default {
  name: "my",
  //   components: {
  //       NavigationBar
  //   },
  data() {
    return {
      input: "",
      characterList: [
      ],
      characterChosen:"",
      final_transcript: "",
      recognizing: false,
      ignore_onend: "",
      start_timestamp: "",
      current_style: "",
      create_email: false,
      recognization: "",
      first_char: /\S/,
      two_line: /\n\n/g,
      one_line: /\n/g,
      uploadFormVisible: false,
      fileList: [],
      uploadForm: {
        title: "",
        desc: "",
      },
      rules: {
        uploadFile: [
          { required: true, message: "Please select file", trigger: "change" },
        ],
        desc: [
          {
            required: true,
            message: "Please input description",
            trigger: "blur",
          },
        ],
      },
    };
  },

  methods: {
    getAllModels(){
      this.$axios({
        method: 'get',
        url: '/models',
      }).then(res => {
        for(var i=0;i<res.data.length;i++){
          console.log(res.data[i])
          var temp={"value":i+1,"label":res.data[i]}

          // console.log(temp)
          this.characterList.push(temp)
        }
        // this.characterChosen=this.characterList[0].label;

      }).catch(error => {
        console.log(error);
        alert("ERROR! Load Models Failed! ");
      });
    },
    send() {
      console.log('send msg.')
      if (this.input != '') {
        var outer_div = document.createElement('div');
        outer_div.style = 'width: 100%; overflow: auto;';
        var div = document.createElement('div');
        div.innerHTML = this.input;
        div.style = 'border: 1px rgb(31, 142, 255) solid; border-radius: 5px; background-color: rgb(31, 142, 255); color: white; float: right; width: fit-content; padding: 6px 10px; margin: 5px; margin-left: 30px;';
        outer_div.append(div);
        document.getElementById("chat-history").append(outer_div);
        document.getElementById("chat-history-box").scrollTop = document.getElementById("chat-history").scrollHeight;

        let params = new FormData();
        params.append("msg", this.input);
        this.input = '';
        this.$axios({
          method: 'post',
          url: '/consult/online',
          data: params
        }).then(res => {
          // window.console.log(res.data);
          outer_div = document.createElement('div');
          outer_div.style = 'width: 100%; overflow: auto;';
          div = document.createElement('div');
          div.innerHTML = res.data;
          div.style = 'border: 1px rgb(235, 237, 240) solid; border-radius: 5px; background-color: rgb(235, 237, 240); float: left; width: fit-content; padding: 6px 10px; margin: 5px; margin-right: 30px;';
          outer_div.append(div);
          document.getElementById("chat-history").append(outer_div);
          setTimeout("testFunction(res)","2000");
          document.getElementById("chat-history-box").scrollTop = document.getElementById("chat-history").scrollHeight;
        }).catch(error => {
          console.log(error);
        });
      }
    },
    showInfo:function (s) {

      if (s) {
        for (var child = document.getElementById("info").firstChild; child; child = child.nextSibling) {
          if (child.style) {
            child.style.display = child.id == s ? 'inline' : 'none';
          }
        }
        document.getElementById("info").setAttribute("style","visibility:'visible'");
      } else {
        document.getElementById("info").setAttribute("style","visibility:'hidden'");
      }
    },
    startButton:function (event) {
      if (this.recognizing) {
        this.recognition.stop();
        return;
      }
      this.final_transcript = '';
      this.recognition.lang = 'en-US';
      this.recognition.start();
      this.ignore_onend = false;
      // document.getElementById("final_span").innerHTML = '';
      // document.getElementById("interim_span").innerHTML = '';
      this.micImgPath='mic-slash.gif';
      // document.getElementById("start_img").setAttribute("src",'../image/mic-slash.gif')  ;
      this.showInfo('info_allow');
      // this.showButtons('none');
      this.start_timestamp = event.timeStamp;
    },
    chooseCharacter(value) {
      console.log(value)
      this.characterChosen=value;
    },
    linebreak: function (s) {
      return s.replace(this.two_line, "<p></p>").replace(this.one_line, "<br>");
    },
    upgrade: function () {
      document
          .getElementById("start_button")
          .setAttribute("style", "visibility :'hidden'");
      this.showInfo("info_upgrade");
    },
    upload() {
      this.uploadFormVisible = false;
      this.$refs.upload.submit();
    },
    openUploadDialog() {
      this.uploadFormVisible = true;
    },
    cancelUpload() {
      this.$refs["uploadForm"].resetFields();
      this.uploadFormVisible = false;
    },
    uploadMedicalArchive(content) {
      // window.console.log(this.fileList.length);
      var imgurl = URL.createObjectURL(content.file);
      window.console.log(imgurl);
      console.log(this.uploadForm.desc);
      window.console.log("this.uploadForm.desc");
      var ques = this.uploadForm.desc;
      let params = new FormData();
      // params.append("title", this.uploadForm.title);
      params.append("desc", this.uploadForm.desc);
      params.append("file", content.file);
      console.log(params)
      // this.loadArchiveList();
      var image = document.createElement("img");
      image.src = URL.createObjectURL(content.file);
      image.style =
          "border: 1px rgb(31, 142, 255) solid; border-radius: 5px; background-color: rgb(31, 142, 255); color: white; float: right; width: 200px; padding: 6px 10px; margin: 5px; margin-left: 30px;"; // document.getElementById("images").appendChild(image);
      var outer_div = document.createElement("div");
      outer_div.style = "width: 100%; overflow: auto;";
      var div = document.createElement("div");
      div.innerHTML = ques;
      div.style =
          "border: 1px rgb(31, 142, 255) solid; border-radius: 5px; background-color: rgb(31, 142, 255); color: white; float: right; width: fit-content; padding: 6px 10px; margin: 5px; margin-left: 30px;";
      outer_div.append(div);
      var outer_div2 = document.createElement("div");
      outer_div.style = "width: 100%; overflow: auto;";
      outer_div2.append(image);
      document.getElementById("chat-history").append(outer_div);
      document.getElementById(
          "chat-history-box"
      ).scrollTop = document.getElementById("chat-history").scrollHeight;

      document.getElementById("chat-history").append(outer_div2);
      document.getElementById(
          "chat-history-box"
      ).scrollTop = document.getElementById("chat-history").scrollHeight;
      // console.log(error);
      console.log("modelname:"+this.characterChosen)
      this.$axios({
        method: "post",
        url: "/archive/user/"+this.characterChosen,
        data: params,
      }).then((res) => {
        window.console.log(res);
        this.$refs["uploadForm"].resetFields();
        // this.uploadFormVisible = false;
        this.$message({
          type: "success",
          message: "Successully uploaded medical archive!",
          showClose: true,
        });
        outer_div = document.createElement("div");
        outer_div.style = "width: 100%; overflow: auto;";
        div = document.createElement("div");
        div.innerHTML = res.data;
        div.style =
            "border: 1px rgb(235, 237, 240) solid; border-radius: 5px; background-color: rgb(235, 237, 240); float: left; width: fit-content; padding: 6px 10px; margin: 5px; margin-right: 30px;";
        outer_div.append(div);
        document.getElementById("chat-history").append(outer_div);
        setTimeout("testFunction(res)", "2000");
        document.getElementById(
            "chat-history-box"
        ).scrollTop = document.getElementById("chat-history").scrollHeight;
      });
    },
    capitalize: function (s) {
      return s.replace(this.first_char, function (m) {
        return m.toUpperCase();
      });
    },
  },
  mounted() {
    this.getAllModels()

    if (!("webkitSpeechRecognition" in window)) {
      this.upgrade();
    } else {
      // document.getElementById("start_button").setAttribute("style","display: 'inline-block'") ;
      this.recognition = new webkitSpeechRecognition();
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      var that = this;
      this.recognition.onstart = function () {
        that.recognizing = true;
        that.showInfo("info_speak_now");
        that.micImgPath = "mic-animate.gif";
        // document.getElementById("start_img").src = '../image/mic-animate.gif';
      };

      this.recognition.onerror = function (event) {
        if (event.error == "no-speech") {
          that.micImgPath = "mic.gif";
          // document.getElementById("start_img").setAttribute("src", '../image/mic.gif');
          that.showInfo("info_no_speech");
          that.ignore_onend = true;
        }
        if (event.error == "audio-capture") {
          that.micImgPath = "mic.gif";
          // document.getElementById("start_img").setAttribute("src",'../image/mic.gif');
          that.showInfo("info_no_microphone");
          that.ignore_onend = true;
        }
        if (event.error == "not-allowed") {
          if (event.timeStamp - start_timestamp < 100) {
            that.showInfo("info_blocked");
          } else {
            that.showInfo("info_denied");
          }
          that.ignore_onend = true;
        }
      };

      this.recognition.onend = function () {
        console.log("endof recog");
        that.recognizing = false;
        if (that.ignore_onend) {
          return;
        }
        that.micImgPath = "mic.gif";
        // document.getElementById("start_img").setAttribute("src", '../image/mic.gif');
        if (!that.final_transcript) {
          that.showInfo("info_start");
          return;
        }
        that.showInfo("");
      };

      this.recognition.onresult = function (event) {
        var interim_transcript = "";
        for (var i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            that.final_transcript += event.results[i][0].transcript;
          } else {
            interim_transcript += event.results[i][0].transcript;
          }
        }
        that.final_transcript = that.capitalize(that.final_transcript);
        // document.getElementById('final_span').setAttribute("innerHTML",this.linebreak(this.final_transcript))  ;
        // document.getElementById('interim_span').setAttribute("innerHTML", this.linebreak(interim_transcript);
        that.input = that.linebreak(that.final_transcript);
        // if (final_transcript || interim_transcript) {
        //   showButtons('inline-block');
        // }
      };
    }
  },
};
</script>

<style scoped>
#consult-div {
  position: fixed;
  top: 25%;
  left: 25%;
  right: 25%;
  bottom: 10%;
  border: 1px rgb(180, 180, 180) solid;
  border-radius: 5px;
}

#chat-div {
  width: 70%;
  height: 100%;
  margin-right: 0;
}

#chat-history-box {
  height: 100%;
  background-color: #fff;
  border: 1px rgb(180, 180, 180) solid;
  border-radius: 5px;
  overflow: auto;
}

#chat-history {
  margin: 15px;
  white-space: pre-line;
}

#input {
  width: 100%;
  height: 150%;
  margin-bottom: 0px;
}

.el-input {
  width: 100%;
}

#info-div {
  width: 32%;
  height: 100%;
  padding-top: 50px;
  text-align: center;
  border-left: 1px rgb(180, 180, 180) solid;
  border-radius: 5px;
  background-color: rgb(245, 247, 250);
}

#name-box {
  margin-top: 20px;
}
#name-box1 {
  margin-top: 10px;
  text-align: left;
  margin-left: 5%;
}

.intro {
  padding-top: 50px;
}
.option-block {
  display: inline-block;
  position: fixed;
  top: 15%;
  /*margin-right: 40px;*/
}

.option-block .el-select {
  width: 400px;
}
#start_button {
  border: 0;
  background-color: transparent;
  padding: 0;
}
</style>
