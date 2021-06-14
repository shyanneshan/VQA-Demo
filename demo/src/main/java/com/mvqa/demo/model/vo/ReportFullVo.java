package com.mvqa.demo.model.vo;

import lombok.Data;

import java.util.Date;

@Data
public class ReportFullVo {

    private Integer id;

    private String name;

    private String date;

    private String state;

    private String data;

    private String classification;

    private String savepath;

    private Integer prec;

    private Integer recall;

    private Integer f1;

    private Float bleu;

    private Integer epoch;

    private Integer batch;
}
