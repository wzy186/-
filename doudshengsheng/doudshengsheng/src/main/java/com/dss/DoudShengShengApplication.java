package com.dss;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("com.dss.mapper")
public class DoudShengShengApplication {
    public static void main(String[] args) {
        SpringApplication.run(DoudShengShengApplication.class, args);
    }
}
