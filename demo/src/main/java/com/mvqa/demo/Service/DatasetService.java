package com.mvqa.demo.Service;

import com.mvqa.demo.Dao.DatasetDao;
import com.mvqa.demo.model.po.DatasetPo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;

@Component
public class DatasetService {
    
    @Autowired
    private DatasetDao datasetDao;

    public ArrayList<DatasetPo> getAllDatasets() {
        return datasetDao.getDatasetPo();


    }
    public void addDataset(DatasetPo datasetPo){
        datasetDao.addDataset(datasetPo);
    }


}
