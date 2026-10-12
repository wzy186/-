import { getDb } from './db.js';
import { REJECT_REASONS } from './constants.js';
import { pathToFileURL } from 'node:url';

/** 值班同事（操作人），登录不做成账号体系，只按姓名区分谁在操作 */
export const OPERATORS = ['林晚晴', '陈默', '苏小满', '赵一凡'];

type Verdict = 'ok' | 'ad' | 'low' | 'attack' | 'rumor' | 'spam';
interface Post { title: string; content: string; img: number }
interface Spec { verdict: Verdict; posts: Post[] }

/** 每种题材给出「典型审核结论」：ok = 应通过，其余对应驳回原因 */
const SPECS: Spec[] = [
  {
    verdict: 'ok',
    posts: [
      {
        title: '【川西小环线7日】保姆级路线分享，附每天住宿推荐',
        content: '国庆前刚走完川西小环线，把路线整理出来给需要的朋友。D1成都出发到四姑娘山，D2双桥沟一整天，D3中路藏寨看碉楼，D4塔公草原到新都桥，D5雅拉雪山徒步，D6折多山回程。全程自驾，油费过路费一共1800左右，住宿旺季人均200一晚。海拔最高4400，红景天提前一周吃，葡萄糖备上。有问题评论区问，知无不言。',
        img: 4,
      },
      {
        title: '刚从色达回来，这条线路真的值得收藏',
        content: '色达开放后一直想去，这次终于成行。佛学院的红房子非常震撼，建议下午四点后上山，光线最好，晚上八点天全黑以后看夜景。注意：天葬台不对游客开放，不要硬闯，尊重当地习俗。县城海拔3800，酒店记得订有制氧机的。全程路况比想象中好，普通轿车也能走，就是弯多，新手慎开夜路。',
        img: 3,
      },
      {
        title: '新疆独库公路全程自驾攻略，一天四季是真的',
        content: '独库公路今年提前通车，赶在封路前跑完了全程。从独山子到库车561公里，最高处铁力买提达坂3200米，山顶真的在飘雪，山下还在穿短袖。加油注意：全线只有乔尔玛、那拉提、巴音布鲁克有加油站，看到加油站就加满。全程限速40-80，不要超速，路上真的有流动测速。',
        img: 3,
      },
      {
        title: '周末逃离计划：杭州周边3个冷门古镇实测',
        content: '受够了乌镇西塘的人山人海，这个月把杭州周边冷门古镇跑了一遍。第一站深澳古镇，几乎没有游客，建筑保存得很好；第二站龙门古镇，孙权故里，雨天人少的时候最有味道；第三站安昌古镇，酱鸭腊肠一条街，吃货必去。三个地方都不收门票，停车免费，适合周末慢慢逛。',
        img: 2,
      },
    ],
  },
  {
    verdict: 'ok',
    posts: [
      {
        title: '我家橘猫的减肥日记：三个月从8.2kg到6.5kg',
        content: '年初体检医生说橘猪超重，肝指标偏高，开始认真减肥。核心就三件事：换低脂处方粮、定时定量喂、每天逗猫棒20分钟。前两周它天天半夜蹲碗边叫，坚持住就熬过来了。三个月减了1.7公斤，现在跑酷都能跳上冰箱了。附上饮食表和体重曲线，有需要的铲屎官可以参考。',
        img: 3,
      },
      {
        title: '捡到一只小奶猫，记录它从450g到现在的成长',
        content: '上个月在小区垃圾桶旁边捡到它，眼睛都没全睁开。带去医院做了检查，除了饿没大毛病。现在一天喂六次羊奶粉，半夜也得起来，真的像带娃。不过看它从爬都爬不稳到现在会追自己尾巴，一切都值了。想收养流浪猫的朋友，一定先去体检驱虫，家里有原住民的要隔离两周。',
        img: 3,
      },
      {
        title: '养了五年的柯基，这些经验新手一定要看',
        content: '柯基的脊椎问题真的不是开玩笑，我家狗子去年髌骨脱位做了手术，花了八千多。教训就是：一定要控制体重、少让它上下楼梯、杜绝后腿站立讨食。另外柯基掉毛量惊人，扫地机器人必备，梳毛要每天。它脾气倔，训练要有耐心，零食奖励比打骂管用一百倍。',
        img: 2,
      },
    ],
  },
  {
    verdict: 'ok',
    posts: [
      {
        title: '巷子里的老面馆，一碗小面吃了十年',
        content: '这家店在沙坪坝老巷子里开了快二十年，没有招牌，全凭熟客带路。二两小面7块钱，麻辣鲜香一样不缺，海椒是老板自己炕的，糊辣壳香得流口水。每天中午十一点开始排队，一点半卖完就关门，周末不营业。老板说不想太累，味道比什么都重要。',
        img: 2,
      },
      {
        title: '测评5家网红火锅，真实感受不吹不黑',
        content: '最近五家刷屏的网红火锅都去吃了，说下真实感受。A家锅底香但是排队两小时起，B家甜品惊艳火锅一般，C家环境好适合拍照味道及格，D家宣传最凶实际最普通，E家社区店反而最惊喜，毛肚新鲜到脆。结论：网红店适合打卡，想认真吃火锅还是找社区老店。',
        img: 4,
      },
      {
        title: '第一次做舒芙蕾就成功的秘诀，其实就三步',
        content: '舒芙蕾看起来高级，其实失败点就三个：蛋白打发程度、翻拌手法、烤箱温度。蛋白打到提起有小弯钩，蛋黄糊和蛋白翻拌别超过20下，烤箱一定要预热充分。附上我的配方和每一步的状态图，按着做基本不会翻车。第一次成功的时候真的很有成就感。',
        img: 3,
      },
    ],
  },
  {
    verdict: 'ok',
    posts: [
      {
        title: '用了三个月，聊聊这把国产机械键盘到底值不值',
        content: '三个月前入手国产98配列，gasket结构，到手649。先说结论：现在国产键盘的做工和手感，真的不输千元级大厂。手感绵软，麻将音很上头，轴体素质稳定。缺点也有：驱动软件做得糙，无线连接偶尔抽风，键帽PBT但字符工艺一般。总体瑕不掩瑜，预算有限的话强烈推荐。',
        img: 2,
      },
      {
        title: '百元蓝牙耳机横评：谁才是性价比之王',
        content: '把市面上百元价位卖得最好的六款蓝牙耳机都买回来了，做了半个月横评。从音质、续航、延迟、通话降噪、佩戴舒适度五个维度打分。结果有点意外：销量最高的两款反而垫底，两个新品牌表现最好。详细评分表放在最后，学生党可以参考。',
        img: 3,
      },
      {
        title: '自己动手给旧笔记本换固态，成本不到300',
        content: '用了五年的笔记本开机要三分钟，本来打算换新，网上查了下发现加个固态就行。SSD两百出头，拆机工具三十，跟着视频教程二十分钟搞定。现在开机十秒，word秒开，还能再战三年。老笔记本卡顿的朋友先别急着换机，加内存换固态是最划算的升级。',
        img: 2,
      },
    ],
  },
  {
    verdict: 'ok',
    posts: [
      {
        title: '出个人闲置 MacBook Air M2，配件发票齐全',
        content: '去年官网购入的MacBook Air M2 16+512，无磕碰无维修，电池循环78次。因为换了新款所以出，发票、原装充电器、盒子都在。同城可面交验机，外地顺丰保价。诚心要的私信，价格可小刀。',
        img: 3,
      },
      {
        title: '搬家清仓：九成新电钢琴低价转，含琴凳琴架',
        content: '罗兰FP-30X，买来一年多，弹了不到五十次，一直罩着防尘罩。搬去外地带不走，出给有缘人。琴凳、琴架、踏板、原装电源全都在，还送两本教材。音色手感都没问题，支持同城上门试琴。',
        img: 3,
      },
      {
        title: '出骑行装备全套：公路车、头盔、码表打包转',
        content: '入坑两年现在退坑，出全套装备。捷安特OCR 5300，M码，成色九新，刚做过保养；GVR头盔L码；迈金码表带踏频器。全部打包出，单买也行。只出同城，欢迎看车试骑。',
        img: 2,
      },
    ],
  },
  {
    verdict: 'ok',
    posts: [
      {
        title: '新买的绿萝叶子发黄，求助各位大神怎么办',
        content: '上个月在花市买的绿萝，最近一周叶子开始发黄，从叶尖开始，有的整片黄掉了。浇水是一周一次，放在阳台散射光的位置。网上说可能是浇水太多烂根，也可能是缺肥。拍了根部的照片，有没有懂的朋友帮忙看看是什么问题，先谢谢了。',
        img: 2,
      },
      {
        title: '求推荐适合新手的咖啡机，预算2k左右',
        content: '平时每天一杯美式，偶尔拿铁，速溶实在喝腻了。预算2000左右，纠结半自动还是胶囊机。看中了三个型号，一个半自动两个胶囊，主要担心半自动清洗麻烦。有使用经验的朋友帮忙分析下，这个预算买哪个不踩坑？',
        img: 1,
      },
      {
        title: '第一次养猫，三个月大的小猫疫苗怎么打？',
        content: '刚领养了三个月大的小奶猫，原主人说还没打过疫苗。问了附近两家宠物医院，一家说打三针妙三多加狂犬，一家说打两针就行，价格也差不少。有点懵，求问这个月龄到底应该怎么打，中间间隔多久，打完后要注意什么？',
        img: 1,
      },
    ],
  },
  {
    verdict: 'ok',
    posts: [
      {
        title: '连续加班一个月，今天终于提离职了',
        content: '在这家公司三年，最近项目上线连续一个月每天十一点下班，今天终于把离职申请交了。不是因为不能吃苦，是看不到任何成长和回报。30岁的年纪裸辞多少有点慌，但继续耗下去更慌。接下来先休息两周，好好想想下一步。同病相怜的朋友，评论区聊聊。',
        img: 1,
      },
      {
        title: '记录转行第三年：从月薪6k到25k的复盘',
        content: '三年前从土木转行做前端，一路走来踩了不少坑。总结下来最重要的三件事：第一，系统学习比刷视频有用，一本红宝书啃三遍胜过收藏一百个视频；第二，早点做项目投简历，面试反馈是最好的学习路径；第三，第一份工作平台比工资重要。详细经历和面试题整理放在后面，希望对想转行的朋友有帮助。',
        img: 1,
      },
      {
        title: '应届生第一份工作，大厂边缘岗还是小厂核心岗？',
        content: '秋招拿了两个offer：大厂的边缘业务岗，总包30w；小厂的核心项目岗，总包22w加期权。身边人都劝我去大厂镀金，但小厂那边面试官聊得很投缘，技术氛围也不错。纠结一周了，各位过来人帮忙出出主意。',
        img: 1,
      },
    ],
  },
  {
    verdict: 'ad',
    posts: [
      {
        title: '诚招打字员，日结300+，在家就能做，宝妈学生优先',
        content: '急招打字录入员若干名，年龄学历不限，会手机就行。主要工作是把扫描件打成电子文档，按件计费，多劳多得，日结300-800。无需押金，扫码进群领任务，名额有限先到先得！',
        img: 1,
      },
      {
        title: '高薪急聘：抖音点赞员，一部手机月入过万',
        content: '在家就能做的兼职来了！给指定视频点赞关注，一单5-20元，熟练后一天能接几十单。手把手教，三分钟上手，当天结算。想赚钱的加V：dianzan8888，备注"兼职"优先通过！',
        img: 1,
      },
      {
        title: '在家兼职月入2万不是梦，导师一对一带你赚钱',
        content: '还在为房贷车贷发愁吗？加入我们副业训练营，专业导师一对一指导，小白也能月入两万。限时免费领课，仅限前100名，点击头像私信"学习"立即加入，错过再等一年！',
        img: 1,
      },
      {
        title: '转发即可赚钱！动动手指日入100+',
        content: '不需要任何技能，只要把我们的广告转发到朋友圈，每条5元，上不封顶。加入推广团队还能拿团队提成，早加入早赚钱，越早加入层级越高。详情加群咨询，群号在主页！',
        img: 1,
      },
    ],
  },
  {
    verdict: 'ad',
    posts: [
      {
        title: '瘦脸针特价999！本月限时优惠，名额仅剩20个',
        content: '医美年中大促！进口瘦脸针原价2999，现特价999，院长亲自注射。安全无痛，效果立竿见影，不满意免费补打。本月活动名额仅剩20个，私信预约锁定优惠价！',
        img: 1,
      },
      {
        title: '三甲医生坐诊，双眼皮手术立减3000',
        content: '本院特邀三甲医院整形科主任坐诊，双眼皮+开眼角套餐立减3000，支持分期付款。术前免费面诊设计，术后专人跟踪回访。暑期特惠倒计时，详情点击主页查看！',
        img: 1,
      },
      {
        title: '牙齿美白体验价99，仅限新客',
        content: '进口冷光美白原价880，新客体验价只要99！一次美白提升三个色阶，全程无痛。牙齿黄的朋友不要错过，数量有限，赶紧私信预约！',
        img: 1,
      },
    ],
  },
  {
    verdict: 'ad',
    posts: [
      {
        title: '免费领试用装，加V：xxx888 先到先得',
        content: '新店开业福利大放送！面膜、精华、身体乳试用装免费领，每人限领一套，邮费自理。数量有限，加V：xxx888 登记地址，先到先得！',
        img: 1,
      },
      {
        title: '内部优惠群，扫码进群领红包，每天都有秒杀',
        content: '购物内部优惠群开放200个名额！群内每天发红包，还有大牌秒杀、漏洞单、隐藏优惠券。进群不需要任何费用，扫码直接进，手慢无！',
        img: 1,
      },
      {
        title: '老粉回馈：关注+转发抽奖，奖品价值5000+',
        content: '为回馈粉丝支持，即日起关注并转发本条内容，即有机会抽取价值5000元的奖品大礼包！中奖率超高，开奖日期见置顶评论。动动手指，说不定就是你！',
        img: 1,
      },
    ],
  },
  {
    verdict: 'ad',
    posts: [
      {
        title: '拼多多砍一刀：还差0.5%就能提现，求帮忙',
        content: '家人们求帮忙砍一刀！已经砍到99.5%了，还差一点点就能提现500块。点我主页置顶链接帮忙砍，砍完私信我，回砍必回！',
        img: 1,
      },
      {
        title: '拼单团购：进口车厘子5斤只要99，还差3人成团',
        content: '产地直发车厘子，JJ级5斤装拼团价只要99，市场价198！现在成团还差3人，点击链接立即参团，人满发货。超时未成团自动退款，放心拼！',
        img: 1,
      },
      {
        title: '9.9抢100元话费，限量秒杀中',
        content: '平台补贴活动，9.9元抢100元话费充值券，限量1000份。点链接进入活动页面，输入手机号即可参与，先到先得，抢完为止！',
        img: 1,
      },
    ],
  },
  {
    verdict: 'low',
    posts: [
      {
        title: '深夜寂寞，找小哥哥视频聊天，懂的来',
        content: '深夜睡不着，有没有小哥哥视频聊天呀？一个人在家太无聊了。想聊的私信我，主动点，别害羞～',
        img: 1,
      },
      {
        title: '同城约会，私信发照片，懂的来',
        content: '周末无聊想找个人一起看电影吃饭，有同城的小哥哥吗？私信互发照片，看对眼就见面。非诚勿扰。',
        img: 1,
      },
      {
        title: '寂寞的夜，想找人陪',
        content: '又是一个人的夜晚，好想有人陪。可以聊天，也可以语音，私信留言，随缘回复。',
        img: 1,
      },
      {
        title: '寻长期饭友，限同城，条件私聊',
        content: '一个人吃饭太无聊，想找个长期饭友。我请客你赏脸，要求见主页置顶。感兴趣私信，条件合适再约时间。',
        img: 1,
      },
    ],
  },
  {
    verdict: 'attack',
    posts: [
      {
        title: '买XX品牌的全是智商税，不服来喷',
        content: '真的受不了了，某品牌除了营销还会什么？同样配置贵一倍，买的人不是傻就是蠢。评论区别洗地，洗一个我喷一个，来啊。',
        img: 0,
      },
      {
        title: '某些球迷素质堪忧，输不起就别看球',
        content: '输球就骂裁判、就砸东西，某些队的球迷就这素质？自己的主队踢得什么样子心里没数吗？还天天在网上带节奏，恶心。',
        img: 0,
      },
      {
        title: '这剧也有人吹？编剧小学毕业了吗',
        content: '最近刷屏的某剧看了两集实在看不下去，逻辑全是漏洞，演技尬出天际。吹这剧的人是不是没看过好东西？评论区来对线，说一个优点我回十个槽点。',
        img: 0,
      },
    ],
  },
  {
    verdict: 'rumor',
    posts: [
      {
        title: '内部消息：某地发生大事，速看速删',
        content: '朋友是内部人员，刚得到的消息，某地出了大事，网上一点风声都没有。这条内容随时会被删，看到的赶紧截图转发给家人朋友。不保证真实，但宁可信其有。',
        img: 0,
      },
      {
        title: '转发这条消息保平安，不转会有厄运',
        content: '大师开光的护身符语，看到即转，转给五个群保全家平安。看到不转，会有不好的事情发生。这不是迷信，是能量场，信不信由你。',
        img: 0,
      },
      {
        title: '千万别买这几种菜！菜贩子从来不吃的秘密',
        content: '菜市场老熟人偷偷告诉我的：这几种菜千万别买，农药超标严重，菜贩子自己都不吃。看完记得转发给家里的长辈，能救一个是一个！',
        img: 0,
      },
    ],
  },
  {
    verdict: 'spam',
    posts: [
      {
        title: '1111111111',
        content: '11111111111111111111',
        img: 0,
      },
      {
        title: '顶顶顶顶顶顶顶',
        content: '顶顶顶顶顶顶顶顶顶顶顶顶顶顶顶',
        img: 0,
      },
      {
        title: '测试勿回',
        content: '这是一条测试内容，请忽略。这是一条测试内容，请忽略。',
        img: 0,
      },
    ],
  },
];

const REASON: Record<Verdict, string | null> = {
  ok: null,
  ad: REJECT_REASONS[0],
  low: REJECT_REASONS[1],
  attack: REJECT_REASONS[2],
  rumor: REJECT_REASONS[3],
  spam: REJECT_REASONS[4],
};

const AUTHORS = [
  '爱旅行的阿May', '程序猿小张', '奶茶三分糖', '山城辣妹子', '键盘侠克星',
  '退休老王', '橘子汽水', '不吃香菜', '凌晨四点的海', '风一样的男子',
  '热心市民刘女士', '数码老白', '铲屎官小李', '干饭第一名', '柠檬不酸',
  '深夜食堂常客', '物理系学渣', '跑步的蜗牛', '巷口理发师', '摸鱼大师兄',
];

/** 固定种子的伪随机，保证每次重置生成的演示数据一致 */
function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hoursAgo(h: number): string {
  return new Date(Date.now() - h * 3600_000).toISOString();
}

/**
 * 生成一对「今天内」的提交/审核时间。
 * 直接按本地自然日零点计算窗口（而不是 hoursAgo(随机)），
 * 保证凌晨（00:00~05:00）首次灌数据时这些记录也落在今天；审核时间晚于提交时间。
 */
function makeTodayTimestamps(rng: () => number): { createdAt: string; reviewedAt: string } {
  const dayStart = new Date();
  dayStart.setHours(0, 0, 0, 0);
  const now = Date.now();
  const latestCreated = now - 5 * 60_000;
  const createdMs =
    latestCreated > dayStart.getTime()
      ? dayStart.getTime() + rng() * (latestCreated - dayStart.getTime())
      : dayStart.getTime() + rng() * Math.max(1, now - dayStart.getTime());
  const reviewedMs = Math.min(createdMs + (0.5 + rng() * 3.5) * 3600_000, now - 60_000);
  return {
    createdAt: new Date(createdMs).toISOString(),
    // 兜底值也封顶到当前时刻，杜绝「审核时间在未来」
    reviewedAt: new Date(Math.max(reviewedMs, Math.min(createdMs + 30_000, now))).toISOString(),
  };
}

/** 清空并重新生成演示数据（npm run seed 也会走到这里） */
export function resetAndSeed(): void {
  const db = getDb();
  db.exec('DELETE FROM review_logs; DELETE FROM submissions; DELETE FROM operators;');

  const insOp = db.prepare('INSERT INTO operators (name) VALUES (?)');
  for (const name of OPERATORS) insOp.run(name);

  const rng = mulberry32(20261012);
  const pool = SPECS.flatMap((s) => s.posts.map((p) => ({ ...p, verdict: s.verdict })));
  // 洗牌后前 PENDING_COUNT 条作为待审核，其余作为已审核历史
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  const PENDING_COUNT = 30;

  const insSub = db.prepare(`
    INSERT INTO submissions
      (title, content, image_urls, author, status, reject_reason, reviewed_by, reviewed_at, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const insLog = db.prepare(`
    INSERT INTO review_logs (submission_id, action, reason, operator, created_at)
    VALUES (?, ?, ?, ?, ?)
  `);

  let imgSeq = 1;
  // 整体包一个事务：避免每条插入各自 autocommit、逐条 fsync
  db.exec('BEGIN');
  try {
    pool.forEach((p, idx) => {
      const author = AUTHORS[Math.floor(rng() * AUTHORS.length)];
      const imageUrls = Array.from(
        { length: p.img },
        () => `https://picsum.photos/seed/mod-${imgSeq++}/600/400`,
      );
      const isPending = idx < PENDING_COUNT;

      if (isPending) {
        // 待审核：最近 72 小时内提交，时间打散，模拟积压
        const createdAt = hoursAgo(rng() * 72);
        insSub.run(p.title, p.content, JSON.stringify(imageUrls), author, 'pending', null, null, null, createdAt);
        return;
      }

      // 已审核历史：前 5 条强制落在今天（保证「今日统计」有数据），其余均匀铺到过去 7 天，让近 7 日图每天都有数据
      const today = idx < PENDING_COUNT + 5;
      const { createdAt, reviewedAt } = today
        ? makeTodayTimestamps(rng)
        : {
            createdAt: hoursAgo((1 + ((idx - PENDING_COUNT - 5) % 7)) * 24 + rng() * 12),
            reviewedAt: '',
          };
      const finalReviewedAt = today
        ? reviewedAt
        : (() => {
            let at = new Date(new Date(createdAt).getTime() + (0.2 + rng() * 4) * 3600_000).toISOString();
            if (new Date(at).getTime() > Date.now()) at = new Date(Date.now() - 5 * 60_000).toISOString();
            return at;
          })();
      const status = p.verdict === 'ok' ? 'approved' : 'rejected';
      const rejectReason = REASON[p.verdict];
      const reviewer = OPERATORS[Math.floor(rng() * OPERATORS.length)];

      const info = insSub.run(
        p.title, p.content, JSON.stringify(imageUrls), author, status, rejectReason,
        reviewer, finalReviewedAt, createdAt,
      );
      const id = Number(info.lastInsertRowid);

      // 少量条目补一条「重新审核」日志，演示误审打回的场景
      if (idx % 8 === 3) {
        const midTime = new Date((new Date(createdAt).getTime() + new Date(finalReviewedAt).getTime()) / 2).toISOString();
        insLog.run(id, 'reopen', null, OPERATORS[Math.floor(rng() * OPERATORS.length)], midTime);
      }
      insLog.run(id, status === 'approved' ? 'approve' : 'reject', rejectReason, reviewer, finalReviewedAt);
    });
    db.exec('COMMIT');
  } catch (e) {
    db.exec('ROLLBACK');
    throw e;
  }

  console.log(`✅ 演示数据已生成：${pool.length} 条投稿（${PENDING_COUNT} 条待审核，${pool.length - PENDING_COUNT} 条已审历史）`);
}

/** 首次启动时数据库为空则自动灌入演示数据 */
export function seedIfEmpty(): boolean {
  const db = getDb();
  const row = db.prepare('SELECT COUNT(*) AS n FROM submissions').get() as unknown as { n: number };
  if (row.n > 0) return false;
  resetAndSeed();
  return true;
}

// npm run seed（直接执行本文件）时强制重置
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  resetAndSeed();
}
