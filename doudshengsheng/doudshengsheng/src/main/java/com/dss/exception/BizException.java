package com.dss.exception;

import lombok.Getter;

/**
 * 业务异常:用于在 service 层抛出可预期的业务错误,由全局处理器转成标准 Result。
 */
@Getter
public class BizException extends RuntimeException {

    private final int code;

    public BizException(String message) {
        super(message);
        this.code = 500;
    }

    public BizException(int code, String message) {
        super(message);
        this.code = code;
    }
}
