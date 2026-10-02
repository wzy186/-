package com.dss.service;

import com.dss.dto.Result;
import com.dss.entity.User;

public interface IUserService {
    Result sendCode(String phone);
    Result login(com.dss.dto.LoginFormDTO form);
    User getById(Long id);
    com.dss.dto.UserDTO toDTO(User user);
}
