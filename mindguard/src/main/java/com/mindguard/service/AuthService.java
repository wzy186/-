package com.mindguard.service;

import com.mindguard.model.Role;
import com.mindguard.model.User;
import com.mindguard.repository.UserRepository;
import com.mindguard.security.JwtUtil;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.Map;

@Service
@RequiredArgsConstructor
public class AuthService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtil jwtUtil;

    public Map<String, Object> register(String username, String password, String role,
                                         String realName, String email, String studentId) {
        if (userRepository.existsByUsername(username)) {
            throw new RuntimeException("用户名已存在");
        }

        User user = User.builder()
                .username(username)
                .password(passwordEncoder.encode(password))
                .role(Role.valueOf(role.toUpperCase()))
                .realName(realName)
                .email(email)
                .studentId(studentId)
                .build();
        userRepository.save(user);

        String token = jwtUtil.generateToken(user.getId(), user.getUsername(), user.getRole().name());
        return Map.of("token", token, "userId", user.getId(),
                "username", user.getUsername(), "role", user.getRole().name());
    }

    public Map<String, Object> login(String username, String password) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("用户不存在"));

        if (!passwordEncoder.matches(password, user.getPassword())) {
            throw new RuntimeException("密码错误");
        }

        String token = jwtUtil.generateToken(user.getId(), user.getUsername(), user.getRole().name());
        return Map.of("token", token, "userId", user.getId(),
                "username", user.getUsername(), "role", user.getRole().name());
    }
}
