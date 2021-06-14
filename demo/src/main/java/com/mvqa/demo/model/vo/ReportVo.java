package com.mvqa.demo.model.vo;

import lombok.Data;

import java.util.Date;

@Data
public class ReportVo {

    private String name;

    private String data;

    private String date;

    private String batchsize;

    private String epoch;

    private String rnn_cell;

    private String embedding;
}
