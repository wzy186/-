package com.dss.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * 统一返回结果
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Result<T> {
    private Boolean success;
    private Integer code;
    private String msg;
    private T data;

    public static <T> Result<T> ok() {
        return new Result<>(true, 200, "success", null);
    }

    public static <T> Result<T> ok(T data) {
        return new Result<>(true, 200, "success", data);
    }

    public static <T> Result<T> fail(String msg) {
        return new Result<>(false, 500, msg, null);
    }

    public static <T> Result<T> fail(Integer code, String msg) {
        return new Result<>(false, code, msg, null);
    }
}
