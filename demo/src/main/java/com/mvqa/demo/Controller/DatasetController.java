package com.mvqa.demo.Controller;

import com.mvqa.demo.Service.DatasetService;
import com.mvqa.demo.model.po.DatasetPo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;

@RestController
@RequestMapping(value="", produces = "application/json;charset=UTF-8")
public class DatasetController {
    @Autowired
    private DatasetService datasetService;

    @GetMapping("/dataSets")
    public ArrayList<DatasetPo> getDatasets(){
        return datasetService.getAllDatasets();
    }
    @GetMapping("/doneLabeling")
    public void doneLabeling(){
    }

    @GetMapping("/dataSetsNames")
    public ArrayList<String> getAllNames(){
        ArrayList<DatasetPo> datasetPos=getDatasets();
        ArrayList<String> ret=new ArrayList<String>();
        for(int i=0;i<datasetPos.size();i++){
            if(datasetPos.get(i).getIslabeled()=="1")//is labeled
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
            datasetPo.setIslabeled("0");
            datasetPo.setStatus(0);
            datasetPo.setTest(test);
            datasetPo.setTrain(train);
            datasetPo.setValid(valid);
        datasetService.addDataset(datasetPo);
        }
}
