package com.mvqa.demo.Controller;

import com.mvqa.demo.Service.DatasetService;
import com.mvqa.demo.model.po.DatasetPo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.util.ArrayList;

@RestController
@RequestMapping(value="", produces = "application/json;charset=UTF-8")
public class DatasetController {
    @Autowired
    private DatasetService datasetService;
    private static final Logger LOGGER = LoggerFactory.getLogger(DatasetController.class);
    @GetMapping("/dataSets")
    public ArrayList<DatasetPo> getDatasets(){
        System.out.println("get datasets.");
        return datasetService.getAllDatasets();
    }
    @GetMapping("/doneLabeling")
    public void doneLabeling(){
    }
    @GetMapping("/dataSetsNamesLabeled")
    public ArrayList<String> getAllNamesLabeld(){
        ArrayList<DatasetPo> datasetPos=getDatasets();
        System.out.println(datasetPos.size());
        for(int i=0;i<datasetPos.size();i++){
            System.out.println(datasetPos.get(i).getName());
        }

        ArrayList<String> ret=new ArrayList<String>();
        for(int i=0;i<datasetPos.size();i++){

            if(datasetPos.get(i).getIslabeled()==0)//is labeled
                continue;
            ret.add(datasetPos.get(i).getName());
        }
        System.out.println(ret);
        return ret;
    }
    @GetMapping("/dataSetsNames")
    public ArrayList<String> getAllNames(){
        ArrayList<DatasetPo> datasetPos=getDatasets();
        System.out.println(datasetPos.size());
        for(int i=0;i<datasetPos.size();i++){
            System.out.println(datasetPos.get(i).getName());
        }

        ArrayList<String> ret=new ArrayList<String>();
        for(int i=0;i<datasetPos.size();i++){

            if(datasetPos.get(i).getIslabeled()==1)//is labeled
                continue;
            ret.add(datasetPos.get(i).getName());
        }
        System.out.println(ret);
        return ret;
    }


    @PostMapping("/addDataset")
    public void addDataset(@RequestParam("name") String name,@RequestParam("description") String description, @RequestParam("train") Integer train, @RequestParam("valid") Integer valid,@RequestParam("test") Integer test) {
            DatasetPo datasetPo = new DatasetPo();
            datasetPo.setName(name);
            datasetPo.setDescription(description);
            datasetPo.setIslabeled(0);
            datasetPo.setStatus(0);
            datasetPo.setTest(test);
            datasetPo.setTrain(train);
            datasetPo.setValid(valid);
        datasetService.addDataset(datasetPo);
        }

    @Value("${dataset.upload}")
    private String uploadPath;
    @Value("${fronted.static}")
    private String imgPath;
    @Value("${dataset.pythonenv}")
    private String pythonEnv;
    @Value("${dataset.QApythonscript}")
    private String pythonScript;

    @PostMapping("/upload")
    @ResponseBody
    @CrossOrigin
    public String upload(@RequestParam("file") MultipartFile file) {
        if (file.isEmpty()) {
            return "Upload failed, please choose a file";
        }

        String fileName = file.getOriginalFilename();

        String filePath = uploadPath;

        File dest = new File(filePath + fileName);
        try {
            file.transferTo(dest);

            python(filePath+fileName);
            LOGGER.info(dest.getName());

            return "success";
        } catch (IOException e) {
            LOGGER.error(e.toString(), e);
        }
        return "Upload failed！";
    }


    public void python(String fileSrc) throws IOException {

        String env = pythonEnv;
        String imgpath = imgPath;
        File dic=new File(".");
        System.out.println(dic.getCanonicalFile());
        String model = pythonScript;
        String cmd=env+" "+model+" --name "+fileSrc + " --imgpath " + imgpath;
        System.out.println(cmd);
        Runtime run=Runtime.getRuntime();
        try{
            Process process=run.exec(cmd);
            InputStream in=process.getInputStream();
            InputStreamReader reader=new InputStreamReader(in);
            BufferedReader br =new BufferedReader(reader);
            StringBuffer sb=new StringBuffer();
            String message;
            while((message=br.readLine())!=null){
                sb.append(message);
            }
            System.out.println(sb);
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }

}
