package com.mvqa.demo.Controller;

//import com.mvqa.demo.Service.MedicalArchiveService;
import com.mvqa.demo.Service.TrainModelService;
        import com.mvqa.demo.entity.PythonCallEntity;
import com.mvqa.demo.model.po.ReportPo;
import org.apache.commons.lang.RandomStringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

        import javax.servlet.http.HttpServletResponse;
        import java.io.*;
import java.util.*;
@RestController
@RequestMapping(value="", produces = "application/json;charset=UTF-8")
public class ConsultController {

    @Autowired
    private PythonCallEntity pythonCallEntity;//=new PythonCallEntity();

    @Autowired
    TrainModelService trainModelService;

    //the train done models [articleNet is excluded.]
    @GetMapping("/models")
    public ArrayList<String> getAllModels(){

        return trainModelService.getAllModels();
}

    //这里改成post图片和问题，调用python回答问题，res是answer,输入是图片的arr和ques
    @PostMapping("/archive/user/{modelName}")
    public Object uploadMedicalArchive(HttpServletResponse response, @RequestParam("desc") String ques, @RequestParam("file") MultipartFile file,
                                       @PathVariable String modelName) throws IOException {
        System.out.println("upload imag to ai robot. Model: "+modelName);
//        System.out.println("upload imag to ai robot. Model: "+modelName);
        // 设置本地保存地址
        String projPath=(new File("")).getCanonicalPath();
        String uploadPath=projPath+"/uploadimag";
        String random = RandomStringUtils.randomAlphanumeric(15);
        // 文件保存url  这是保留前台上传文件的后缀(也可以在这里做文件格式的效验)
        File path = new File(uploadPath, random + file.getOriginalFilename().substring(file.getOriginalFilename().length() - 4));
        // 判断父级目录是否存在
        if (!path.getParentFile().exists()) {
            path.getParentFile().mkdir();
        }
        // 文件对拷
        try {
            file.transferTo(path);
            // 成功返回前台数据
            System.out.println( uploadPath+'/'+ path.getName());
//            return folder+'/'+ path.getName();
        } catch (IOException e) {
            e.printStackTrace();
        }
        String ans="ans null";
        Runtime run = Runtime.getRuntime();
        String imgpath=uploadPath+'/'+ path.getName();
        String question=ques;
        ReportPo reportPo=trainModelService.getReportPo(modelName);
        String savePath=(new File("")).getCanonicalPath()+"/weights";
        switch (reportPo.getClassification()){
            case "VGG-Seq2Seq":
                System.out.println("will predict seq2seq.");
                //call vgg seq2seq
                try{
                    ans=pythonCallEntity.VGG_Seq2Seq_predict(reportPo.getName(), reportPo.getData(),
                            imgpath, question , reportPo.getSavepath());
                    System.out.println(ans);
//                    ReportPo reportPo1_vgg=trainModelService.updateReportBleuByName(reportPo.getName(),Float.parseFloat(res));

                }catch (Exception e){
                    e.printStackTrace();
                    trainModelService.setTrainError(reportPo.getName());
                }
                break;
            case "NLM":
                System.out.println("will predict nlm.");
                //call NLM
                try{
                    ans=pythonCallEntity.NLM_predict( reportPo.getName(), question, imgpath,reportPo.getData(),reportPo.getSavepath(),reportPo.getBatchsize());
//                    ReportPo reportPo1_nlm=trainModelService.updateReportBleuByName(reportPo.getName(),Float.parseFloat(res));
                }catch (Exception e){
                    e.printStackTrace();
                    trainModelService.setTrainError(reportPo.getName());
                }
                break;
            case "ODL":
                System.out.println("will predict odl.");

                try{
                    String ae="";
                    String maml="";
                    if(reportPo.getConstructor().equals("both")){
                        ae="True";
                        maml="True";
                    }else if(reportPo.getConstructor().equals("maml")){
                        ae="False";
                        maml="True";
                    }else if(reportPo.getConstructor().equals("none")){
                        ae="False";
                        maml="False";
                    }else {
                        ae="True";
                        maml="False";
                    }

                    ans=pythonCallEntity.ODL_predict(reportPo.getName(),question,imgpath,reportPo.getSavepath(),
                            reportPo.getAttention(),reportPo.getRnnCell(),ae,maml,reportPo.getData());
                }catch (Exception e){
                    e.printStackTrace();
                    trainModelService.setTrainError(reportPo.getName());
                }
                break;
            case "MMBERT":
                System.out.println("will predict mmbert.");
                //call mmbert python
                ans=pythonCallEntity.MMBERT_predict(reportPo.getName(),question,imgpath,reportPo.getSavepath(),reportPo.getData());
                break;
//            case "ArticleNet":
            // can't predict articleNet because of its character.
//                System.out.println("will train ArticleNet.");
//                //call knowledge embedding
//                break;
            default:
                System.out.println("this model unsupported.");
                return "fail.";
        }
        return ans;
//        String model=String.valueOf(3);
//        String cmdStr = "/home/lf/anaconda3/envs/tf/bin/python3.6 /home/lf/桌面/vqacode/NeuralNetwork-ImageQA-master/question_answer.py --image="+imgpath+" --question="+question+" --model=2";
//        try {
//            Process process = run.exec(cmdStr);
//            InputStream in = process.getInputStream();
//            InputStreamReader reader = new InputStreamReader(in);
//            BufferedReader br = new BufferedReader(reader);
//            StringBuffer sb = new StringBuffer();
//            String message;
//            while((message = br.readLine()) != null) {
//                sb.append(message);
//            }
//            System.out.println(sb);
//            return sb;
//        } catch (IOException e) {
//            // TODO Auto-generated catch block
//            e.printStackTrace();
//        }
//        return "";
    }

}
