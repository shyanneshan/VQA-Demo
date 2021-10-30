package com.mvqa.demo.Service;

import com.mvqa.demo.entity.CSVEntity;
import com.mvqa.demo.model.po.ReportPo;
import com.mvqa.demo.model.vo.ReportFullVo;
import org.apache.commons.lang.RandomStringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.mvqa.demo.Dao.TrainModelDao;

import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;


@Service
public class TrainModelService {
    @Autowired
    TrainModelDao trainModelDao;
    /**
    * @author     ：hybai
    * @date       ：Created in
    * @description：TODO
    */
    public ReportPo getReportPo(String modelName){
        return trainModelDao.getReportPo(modelName);
    }

    public void deleteFailed(){
        trainModelDao.deleteFailed();
    }
    /**
     * @author ：hybai
     * @date ：Created in
     * @description：TODO
     */
    public ArrayList<String> getAllModels() {
        ArrayList<ReportPo> reportPos = trainModelDao.getAllModels();
        ArrayList<String> allModels=new ArrayList<String>();
        for (int idx = 0; idx < reportPos.size(); idx++) {
            if (reportPos.get(idx).getClassification()=="ArticleNet")
                continue;
            allModels.add(reportPos.get(idx).getName());
            System.out.println(reportPos.get(idx).getName());
//            if(reportPos.get(idx).getState()=="done"){
//                allModels.add(reportPos.get(idx).getName());
//                System.out.println(reportPos.get(idx));
//            }

        }
        return allModels;
    }

    public ArrayList<String> getAllModelNames_exist() {
        ArrayList<ReportPo> reportPos = trainModelDao.getAllModels_exist();
        ArrayList<String> allModels=new ArrayList<String>();
        for (int idx = 0; idx < reportPos.size(); idx++) {
            allModels.add(reportPos.get(idx).getName());
//            System.out.println(reportPos.get(idx).getName());

        }
        return allModels;
    }

    public ArrayList<ReportPo> getAllDonePos(){
        return trainModelDao.getAllModels();
    }

    /**
     * @author ：hybai
     * @date ：Created in
     * @description：TODO
     */
    public ArrayList<ReportFullVo> getAllReport() {
        return trainModelDao.getAllReport();
    }

    /**
     * @author ：hybai
     * @date ：Created in
     * @description：Update one report (get bleu
     */
    public ReportPo updateReportBleuByName(String name, Float bleu,String savepath) {
        ReportPo reportPo = trainModelDao.updateReportBleuByName(name, bleu, savepath);
        return reportPo;
    }

    /**
     * @author ：hybai
     * @date ：Created in
     * @description：Update one report (get precision recall f1
     */
    public ReportPo updateReportByName(String name, Integer precision, Integer recall, Integer f1, String savepath) {
        ReportPo reportPo = trainModelDao.updateReportByName(name, precision, recall, f1, savepath);
        return reportPo;
    }

    /**
     * @author ：hybai
     * @date ：Created in
     * @description：insert a new report when train a new model
     */
    public ReportPo insertReport(String name, String data, String model, String epoch,
                                 String batchsize, String rnn_cell, String embedding,
                                 String attention, String constructor) throws ParseException {
        Integer epoch_ = Integer.parseInt(epoch);
        Integer batchsize_ = Integer.parseInt(batchsize);
        ReportPo reportPo = trainModelDao.insertReport(name, data, model, epoch_, batchsize_, rnn_cell, embedding,attention,constructor);
        return reportPo;
    }

    /**
     * @author ：hybai
     * @date ：Created in
     * @description：process error created when trainning
     */
    public void setTrainError(String name) {
        trainModelDao.setTrainError(name);
    }

    @Autowired
    private CSVEntity csvEntity;

    public String createReports() throws IOException {

//        CSVEntity csvEntity=new CSVEntity(RandomStringUtils.randomAlphanumeric(10));
        ArrayList<ReportPo> reportPos=trainModelDao.getAllModels();
        System.out.println(reportPos);
        csvEntity.setDestpath();
        return csvEntity.write2csv(reportPos);
    }
}
