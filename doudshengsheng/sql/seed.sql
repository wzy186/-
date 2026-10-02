-- 兜省省 默认数据种子
-- 用法:mysql -uroot doudshengsheng < sql/seed.sql
-- 可重复执行(先 DELETE 再 INSERT)
USE doudshengsheng;

-- ============ 1. 用户昵称正经化 ============
UPDATE tb_user SET nick_name='省钱小能手', icon='' WHERE id=1;
UPDATE tb_user SET nick_name='薅羊毛大队长', icon='' WHERE id=2;
UPDATE tb_user SET nick_name='探店达人', icon='' WHERE id=3;
UPDATE tb_user SET nick_name='奶茶控', icon='' WHERE id=4;
UPDATE tb_user SET nick_name='吃货老王', icon='' WHERE id=5;
UPDATE tb_user SET nick_name='K歌之王', icon='' WHERE id=6;
UPDATE tb_user SET nick_name='美甲小姐姐', icon='' WHERE id=7;
UPDATE tb_user SET nick_name='夜猫子酒店客', icon='' WHERE id=8;
UPDATE tb_user SET nick_name='烤肉爱好者', icon='' WHERE id=9;
UPDATE tb_user SET nick_name='影迷小张', icon='' WHERE id=10;

-- 管理员:用户1(role=1)可登录商户后台 /admin
UPDATE tb_user SET role=1 WHERE id=1;

-- ============ 2. 商铺扩充到 12 家(已有 1-8,补 9-12)============
INSERT INTO tb_shop (id, name, type_id, area, address, x, y, avg_price, sold, comments, score) VALUES
(9,'特价烧烤摊',1,'海淀区','中关村大街27号',116.316000,39.990000,5500,380,90,89),
(10,'欢乐电玩城',2,'朝阳区','朝阳路67号',116.490000,39.930000,8800,150,50,82),
(11,'省钱洗车行',4,'海淀区','学院路12号',116.350000,39.995000,3500,680,120,90),
(12,'平价理发铺',4,'朝阳区','工体北路',116.440000,39.935000,3800,420,90,86)
ON DUPLICATE KEY UPDATE name=VALUES(name);

-- ============ 3. 优惠券(补几张大券)============
INSERT INTO tb_voucher (id, shop_id, title, sub_title, pay_value, actual_value, type, status) VALUES
(20, 1, '满20减8奶茶券', '每日可用', 0, 800, 1, 1),
(21, 2, '满100减30火锅券', '工作日通用', 0, 3000, 1, 1),
(22, 6, '满150减50烤肉券', '周末专享', 0, 5000, 1, 1),
(23, 7, '19.9元秒杀电影票', '限量抢购', 1990, 4500, 2, 1)
ON DUPLICATE KEY UPDATE title=VALUES(title);

-- 秒杀券附加:voucher 23,库存 50
INSERT INTO tb_seckill_voucher (voucher_id, stock, begin_time, end_time) VALUES
(23, 50, '2024-01-01 00:00:00', '2099-12-31 23:59:59')
ON DUPLICATE KEY UPDATE stock=VALUES(stock);

-- ============ 4. 探店笔记(正经内容,关联商铺和用户)============
DELETE FROM tb_blog;
INSERT INTO tb_blog (id, shop_id, user_id, title, content, liked, comments) VALUES
(1, 1, 3, '兜省省奶茶铺,10块钱喝出星巴克的感觉', '今天路过中关村,发现这家奶茶铺正在搞活动,满20减8,相当于一杯只要7块!奶盖茶特别浓郁,推荐试试。', 86, 12),
(2, 2, 5, '省钱火锅店隐藏菜单曝光', '这家火锅店工作日满100减30,人均不到60就能吃到撑。毛肚、鸭肠必点,锅底是牛油的,越煮越香。', 142, 28),
(3, 7, 10, '一折电影城,9块9看新片', '朝阳大悦城这家影城常年有秒杀,19.9的电影票比会员还便宜。IMAX厅也能用,周末早点去抢。', 203, 45),
(4, 6, 9, '半价烤肉馆,人均50吃到扶墙', '五道口这家烤肉馆周末满150减50,五花肉和牛舌是招牌,自己烤更有感觉。记得提前排号。', 98, 19),
(5, 3, 6, '薅羊毛KTV,下午场6折', '学院路这家KTV工作日下午场6折,人均30唱一下午。包间干净,音响也不错,适合团建。', 67, 8),
(6, 4, 7, '白菜价美甲,三里屯的隐藏宝藏', '三里屯这家美甲店经常有团购,基础款才38,做工比商场里精细。小姐姐手艺很好,推荐法式。', 54, 11),
(7, 8, 1, '特惠理发店,15块剪出造型', '苏州街这家理发店新客15块,老师傅手艺扎实,不推销办卡。男生剪短发首选。', 39, 6),
(8, 5, 8, '骨折价酒店,出差党的福音', '西二环这家酒店常年有特惠房,200出头住标间,含早。位置好,地铁直达,出差首选。', 76, 14);

-- ============ 5. 关注关系(用户1关注3/5/10,用户2关注3/5,用户4关注3)============
DELETE FROM tb_follow;
INSERT INTO tb_follow (user_id, follow_user_id) VALUES
(1, 3), (1, 5), (1, 10),
(2, 3), (2, 5),
(4, 3),
(6, 6), (7, 7);

-- ============ 6. 红包雨:不在 SQL 里插,由后端启动时 DataInitializer 自动创建可抢的场次 ============
DELETE FROM tb_red_packet;

SELECT '种子数据已加载' AS result;
SELECT CONCAT('商铺 ', COUNT(*), ' 家') AS s FROM tb_shop
UNION ALL SELECT CONCAT('笔记 ', COUNT(*), ' 篇') FROM tb_blog
UNION ALL SELECT CONCAT('关注 ', COUNT(*), ' 条') FROM tb_follow
UNION ALL SELECT CONCAT('优惠券 ', COUNT(*), ' 张') FROM tb_voucher;
