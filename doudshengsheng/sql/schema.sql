-- 兜省省 数据库
CREATE DATABASE IF NOT EXISTS doudshengsheng DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE doudshengsheng;

-- 用户表
DROP TABLE IF EXISTS tb_user;
CREATE TABLE tb_user (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
    phone VARCHAR(11) NOT NULL COMMENT '手机号',
    password VARCHAR(128) DEFAULT '' COMMENT '密码(本项目用验证码登录,留空)',
    nick_name VARCHAR(32) DEFAULT '' COMMENT '昵称',
    icon VARCHAR(255) DEFAULT '' COMMENT '头像',
    role TINYINT DEFAULT 0 COMMENT '0普通 1管理员',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_phone (phone)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表';

-- 商铺类型
DROP TABLE IF EXISTS tb_shop_type;
CREATE TABLE tb_shop_type (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    name VARCHAR(32) NOT NULL,
    sort INT DEFAULT 0,
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='商铺类型';

-- 商铺表
DROP TABLE IF EXISTS tb_shop;
CREATE TABLE tb_shop (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    name VARCHAR(64) NOT NULL,
    type_id BIGINT UNSIGNED NOT NULL,
    images VARCHAR(1024) DEFAULT '',
    cover VARCHAR(255) DEFAULT '' COMMENT '封面图URL',
    area VARCHAR(32) DEFAULT '' COMMENT '大区',
    address VARCHAR(128) DEFAULT '',
    x DECIMAL(10,7) DEFAULT NULL COMMENT '经度',
    y DECIMAL(10,7) DEFAULT NULL COMMENT '纬度',
    avg_price BIGINT DEFAULT 0 COMMENT '均价(分)',
    sold INT DEFAULT 0 COMMENT '销量',
    comments INT DEFAULT 0 COMMENT '评论数',
    score INT DEFAULT 0 COMMENT '评分(0-100)',
    open_hours VARCHAR(64) DEFAULT '',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_type (type_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='商铺表';

-- 优惠券
DROP TABLE IF EXISTS tb_voucher;
CREATE TABLE tb_voucher (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    shop_id BIGINT UNSIGNED DEFAULT NULL COMMENT '商铺id,null=全场券',
    title VARCHAR(64) NOT NULL,
    sub_title VARCHAR(64) DEFAULT '',
    rules VARCHAR(255) DEFAULT '',
    pay_value BIGINT DEFAULT 0 COMMENT '支付价值(分)',
    actual_value BIGINT DEFAULT 0 COMMENT '抵扣价值(分)',
    type TINYINT DEFAULT 1 COMMENT '1普通券 2秒杀券',
    status TINYINT DEFAULT 1 COMMENT '1上架 0下架',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_shop (shop_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='优惠券';

-- 秒杀券附加信息
DROP TABLE IF EXISTS tb_seckill_voucher;
CREATE TABLE tb_seckill_voucher (
    voucher_id BIGINT UNSIGNED NOT NULL COMMENT '优惠券主键',
    stock INT NOT NULL COMMENT '库存',
    begin_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (voucher_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='秒杀券附加信息';

-- 优惠券订单
DROP TABLE IF EXISTS tb_voucher_order;
CREATE TABLE tb_voucher_order (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键(全局唯一id)',
    user_id BIGINT UNSIGNED NOT NULL,
    voucher_id BIGINT UNSIGNED NOT NULL,
    pay_type TINYINT DEFAULT 1 COMMENT '1余额 2微信',
    status TINYINT DEFAULT 1 COMMENT '1未支付 2已支付 3已核销 4已取消',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_user_voucher (user_id, voucher_id) COMMENT '一人一单约束',
    KEY idx_voucher (voucher_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='优惠券订单';

-- 博客(探店笔记)
DROP TABLE IF EXISTS tb_blog;
CREATE TABLE tb_blog (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    shop_id BIGINT UNSIGNED DEFAULT NULL,
    user_id BIGINT UNSIGNED NOT NULL,
    title VARCHAR(128) NOT NULL,
    content TEXT,
    images VARCHAR(1024) DEFAULT '',
    liked INT DEFAULT 0,
    comments INT DEFAULT 0,
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='探店笔记';

-- 关注关系
DROP TABLE IF EXISTS tb_follow;
CREATE TABLE tb_follow (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    user_id BIGINT UNSIGNED NOT NULL COMMENT '关注者',
    follow_user_id BIGINT UNSIGNED NOT NULL COMMENT '被关注者',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_follow (user_id, follow_user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='关注关系';

-- 红包
DROP TABLE IF EXISTS tb_red_packet;
CREATE TABLE tb_red_packet (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键(全局唯一id)',
    title VARCHAR(64) NOT NULL,
    total_amount BIGINT NOT NULL COMMENT '总金额(分)',
    count INT NOT NULL COMMENT '总个数',
    remain_count INT NOT NULL COMMENT '剩余个数',
    got_count INT DEFAULT 0 COMMENT '已领个数',
    status TINYINT DEFAULT 1 COMMENT '1进行中 2已抢完 3已退款关闭',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME DEFAULT NULL COMMENT '结束时间',
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='红包雨场次';

-- 红包领取记录
DROP TABLE IF EXISTS tb_red_packet_record;
CREATE TABLE tb_red_packet_record (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    red_packet_id BIGINT UNSIGNED NOT NULL,
    user_id BIGINT UNSIGNED NOT NULL,
    amount BIGINT NOT NULL COMMENT '领取金额(分)',
    grab_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_rp_user (red_packet_id, user_id) COMMENT '一人一次',
    KEY idx_rp (red_packet_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='红包领取记录';

-- ============ 测试数据 ============
INSERT INTO tb_shop_type (id, name, sort) VALUES
(1,'美食',1),(2,'娱乐',2),(3,'丽人',3),(4,'生活服务',4),(5,'酒店',5);

INSERT INTO tb_shop (id, name, type_id, area, address, x, y, avg_price, sold, comments, score) VALUES
(1,'兜省省奶茶铺',1,'海淀区','中关村大街1号',116.310003,39.991956,1500,320,80,90),
(2,'省钱火锅店',1,'朝阳区','建国路88号',116.481028,39.995000,8800,560,200,95),
(3,'薅羊毛KTV',2,'海淀区','学院路30号',116.352000,40.000000,6800,120,40,80),
(4,'白菜价美甲店',3,'朝阳区','三里屯路19号',116.454000,39.940000,3800,210,60,85),
(5,'骨折价酒店',5,'海淀区','西二环中路',116.330000,39.970000,28800,90,30,88),
(6,'半价烤肉馆',1,'海淀区','五道口',116.338000,39.992000,9900,420,110,92),
(7,'一折电影城',2,'朝阳区','朝阳大悦城',116.510000,39.925000,4500,680,260,93),
(8,'特惠理发店',4,'海淀区','苏州街',116.312000,39.980000,3800,500,150,87);

INSERT INTO tb_voucher (id, shop_id, title, sub_title, pay_value, actual_value, type, status) VALUES
(1, 1, '满10减5奶茶券', '每日一杯更省钱', 0, 500, 1, 1),
(2, 2, '8折火锅券', '工作日通用', 0, 880, 1, 1),
(3, NULL, '全场满100减20', '兜省省新人专享', 0, 2000, 1, 1),
(10, 7, '9.9元秒杀电影票', '限量抢购', 990, 4500, 2, 1);

INSERT INTO tb_seckill_voucher (voucher_id, stock, begin_time, end_time) VALUES
(10, 100, '2024-01-01 00:00:00', '2099-12-31 23:59:59');

INSERT INTO tb_red_packet (id, title, total_amount, count, remain_count, got_count, status) VALUES
(0, 'init', 0, 0, 0, 0, 1);
DELETE FROM tb_red_packet WHERE id = 0;
