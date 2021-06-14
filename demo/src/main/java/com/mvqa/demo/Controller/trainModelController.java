package com.mvqa.demo.Controller;

import com.mvqa.demo.Service.TrainModelService;
import com.mvqa.demo.entity.PythonCallEntity;
import com.mvqa.demo.model.po.ReportPo;
import com.mvqa.demo.model.vo.ReportFullVo;
import com.mvqa.demo.model.vo.ReportVo;
import io.swagger.annotations.Api;
import org.openqa.selenium.WebDriver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.bind.annotation.*;


import javax.servlet.ServletException;
import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.net.URLEncoder;
import java.text.ParseException;
import java.util.ArrayList;

@Configuration
@Api(value = "send data to train and get result")
@RestController
@RequestMapping(value="", produces = "application/json;charset=UTF-8")
public class trainModelController {
    @Autowired
    TrainModelService trainModelService;

    PythonCallEntity pythonCallEntity=new PythonCallEntity();

    String downloadpath="/home/wxl/Documents/VQADEMO/download";

    @GetMapping("/allDatasets")
    public ArrayList<String> getallDatasets()  {
//        ArrayList<ReportFullVo> reportFullVos=trainModelService.getAllReport();
//        for(int idx=0;idx<reportFullVos.size();idx++){
//
//        }
        ArrayList<String> ret=new ArrayList<String>();
        ret.add("VQA-Med-2019");
        ret.add("VQA-RAD");
        ret.add("MVQA");
        return ret;
//        return trainModelService.getAllReport();
    }
    //error train set status "failed"
    @RequestMapping(value = "/error/{name}", method = RequestMethod.POST)
    public void setTrainError(HttpServletRequest request,@PathVariable("name") String name) throws IOException, ParseException
    {
        trainModelService.setTrainError(name);
    }

    //    @PostMapping("/train/{model}")
    @RequestMapping(value = "/train/{model}", method = RequestMethod.POST)
    public String insertReport(HttpServletRequest request,@RequestBody ReportVo reportVo,
                               @PathVariable("model") String model) throws IOException, ParseException
    {
        //first,name data date state=running ----------insert to db
        //second,train ------------listening for the end of training
        //third,return savepath,Precision,Recall,f1-score
        ReportPo reportPo=trainModelService.insertReport(reportVo.getName(),reportVo.getData(),model,
                reportVo.getEpoch(),reportVo.getBatchsize(),reportVo.getRnn_cell(),reportVo.getEmbedding());
        //switch choose model
        String cmd="";
        String savePath=(new File("")).getCanonicalPath()+"/weights";
        switch (model){
            case "VGG-Seq2Seq":
                System.out.println("will train seq2seq.");
                //call vgg seq2seq
                try{
                    String res=pythonCallEntity.VGG_Seq2Seq_train("train",reportPo.getData(),reportPo.getName(),
                            String.valueOf(reportPo.getEpoch()),String.valueOf(reportPo.getBatchsize()),savePath);
                    System.out.println(res);
                    ReportPo reportPo1_vgg=trainModelService.updateReportBleuByName(reportPo.getName(),Float.parseFloat(res));

                }catch (Exception e){
                    e.printStackTrace();
                    trainModelService.setTrainError(reportPo.getName());
                }
                break;
            case "NLM":
                System.out.println("will train nlm.");
                //call NLM
                String w2v="";
                String glove="";
                if(reportPo.getEmbedding()=="w2v"){
                    w2v="true";
                    glove="false";
                }else if(reportPo.getEmbedding()=="glove"){
                    w2v="false";
                    glove="true";
                }else {
                    w2v="false";
                    glove="false";
                }
                try{
//                    String savePath=(new File("")).getCanonicalPath()+"/weights";
                    String res=pythonCallEntity.NLM_train("train",reportPo.getData(),reportPo.getName(),String.valueOf(reportPo.getEpoch()),
                            String.valueOf(reportPo.getBatchsize()),glove,w2v,reportPo.getRnnCell(),savePath);
                    System.out.println(res);
                    ReportPo reportPo1_nlm=trainModelService.updateReportBleuByName(reportPo.getName(),Float.parseFloat("0"));
                }catch (Exception e){
                    e.printStackTrace();
                    trainModelService.setTrainError(reportPo.getName());
                }
                break;
            case "CR":
                System.out.println("will train cr.");
                //call CR
                break;
            case "ODL":
                System.out.println("will train odl.");
                //call ODL
                break;
            case "MMBERT":
                System.out.println("will train mmbert.");
                //call mmbert python
                try{
                    String res=pythonCallEntity.MMBERT("train",reportPo.getData(),reportPo.getName(),
                            String.valueOf(reportPo.getEpoch()),String.valueOf(reportPo.getBatchsize()));
                    System.out.println(res);
//                    ReportPo reportPo1_nlm=trainModelService.updateReportByName(reportPo.getName(),Float.parseFloat("0"));
                }catch (Exception e){
                    e.printStackTrace();
                    trainModelService.setTrainError(reportPo.getName());
                }

                break;
//            case "CGMVQA":
//                System.out.println("will train cgm.");
//                //call cgm python

//                break;
            case "Knowledge Embedded Metalearning":
                System.out.println("will train knowledge.");
                //call knowledge embedding
                break;
            default:
                System.out.println("this model unsupported.");
                return "fail.";
        }
        return "";
    }

    /**
     * @author     ：hybai
     * @date       ：Created in
     * @description：when  enter sys ,show all reports
     */
    @GetMapping("/reports")
    public ArrayList<ReportFullVo> getReports(HttpServletRequest request)  {
//        ArrayList<ReportFullVo> reportFullVos=trainModelService.getAllReport();
//        for(int idx=0;idx<reportFullVos.size();idx++){
//
//        }
        return trainModelService.getAllReport();
    }

        /**
         * @author     ：hybai
         * @date       ：Created in
         * @description：download reports
         */
        private WebDriver driver;
        @GetMapping(value = "/downloadReports")
        public String downloadFile(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {

//            FirefoxOptions options = new FirefoxOptions();
//            options.addPreference("browser.download.folderList", 2);
//            options.addPreference("browser.download.dir", "/home/wxl");
//            options.addPreference("browser.download.useDownloadDir", true);
//            options.addPreference("browser.helperApps.neverAsk.saveToDisk", "application/octet-stream,"
//                    + " application/vnd.ms-excel, text/csv, application/zip,application/exe");
//            System.setProperty("webdriver.firefox.marionette",(new File("")).getCanonicalPath()+"/demo/src/main/resources/geckodriver-v0.9.0-linux64/geckodriver");
//            driver = new FirefoxDriver(options);
            //1.要获取下载文件的路径
        /*String realPath = this.getServletContext().getRealPath("/1.png");*/
        String filepath=trainModelService.createReports();
        String realPath =filepath;
        System.out.println("下载文件的路径:" + realPath);
        //2.下载的文件名称是啥    这个是截取文件名的方法()
        //realPath.lastIndexOf("\\") + 1:这是生成一个索引,从后往前找找到"\\"的位置索引,
        //substring():这是根据索引截取索引后面的字符串
        String filename = realPath.substring(realPath.lastIndexOf("\\") + 1);
        //3.设置想办法让浏览器能够支持下载我们需要的东西
        resp.setHeader("Content-Disposition","attachment;filename="+ URLEncoder.encode(filename,"UTF-8"));


        //4.获取下载文件的输入流 (就是把我们的文件变成流)
        FileInputStream in = new FileInputStream(realPath);
        System.out.println(in);
        //5. 创建文件缓冲区
        int len = 0;
        byte[] buffer = new byte[1024];
        //6. 获取FileOutPutStream对象
        ServletOutputStream outputStream = resp.getOutputStream();
        //6. 将FileOutPutStream流写入到buff缓冲区,使用OutPutStream将缓冲区的数据输出到客户端!
        while ((len = in.read(buffer))>0){
            outputStream.write(buffer,0,len);
        }
        in.close();
        outputStream.close();
        return realPath;
        }


}

