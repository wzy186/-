package com.dss.controller;

import com.dss.dto.LoginFormDTO;
import com.dss.dto.Result;
import com.dss.entity.User;
import com.dss.service.IUserService;
import com.dss.utils.UserHolder;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Pattern;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

@Tag(name = "用户", description = "短信验证码登录、当前用户")
@RestController
@RequestMapping("/user")
@RequiredArgsConstructor
@Validated
public class UserController {

    private final IUserService userService;

    /**
     * 发送验证码
     */
    @Operation(summary = "发送短信验证码")
    @PostMapping("/code")
    public Result sendCode(@RequestParam("phone")
                           @Pattern(regexp = "^1\\d{10}$", message = "手机号格式错误") String phone) {
        return userService.sendCode(phone);
    }

    /**
     * 登录(验证码)
     */
    @Operation(summary = "登录/注册", description = "验证码登录,Token 存 Redis Hash,有效期 30 分钟")
    @PostMapping("/login")
    public Result login(@RequestBody @Valid LoginFormDTO form) {
        return userService.login(form);
    }

    /**
     * 当前登录用户
     */
    @Operation(summary = "获取当前登录用户")
    @GetMapping("/me")
    public Result me() {
        return Result.ok(UserHolder.getUser());
    }

    /**
     * 查询用户(测试用)
     */
    @Operation(summary = "按 ID 查询用户")
    @GetMapping("/{id}")
    public Result queryById(@Parameter(description = "用户 ID") @PathVariable Long id) {
        User user = userService.getById(id);
        return Result.ok(userService.toDTO(user));
    }
}
