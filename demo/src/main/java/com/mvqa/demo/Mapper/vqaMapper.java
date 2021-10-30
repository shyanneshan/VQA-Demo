package com.mvqa.demo.Mapper;

import org.apache.ibatis.annotations.Mapper;
import org.springframework.stereotype.Repository;
import com.mvqa.demo.entity.Photo;
import java.util.ArrayList;

@Mapper
@Repository
public interface vqaMapper {
    ArrayList<Photo> getAllPatients(String flag);
}
