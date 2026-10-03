import { describe, expect, it } from 'vitest';
import { mask } from '../src/lib/mask';

describe('看板脱敏', () => {
  it('手机号只留前三后四', () => {
    expect(mask('手机 13812345678，您联系他')).toBe('手机 138****5678，您联系他');
  });
  it('门牌地址整段隐藏', () => {
    expect(mask('地址是朝阳区某某小区 3 号楼 502')).toContain('［地址已隐藏］');
  });
  it('邮箱只留首字母', () => {
    expect(mask('联系 zhangming@example.com')).toBe('联系 z***@example.com');
  });
  it('拦截下来的原句进看板后不含完整手机号', () => {
    const raw = '签收人是张明，手机 13812345678，地址是朝阳区某某小区 3 号楼 502';
    expect(mask(raw)).not.toContain('13812345678');
  });
  it('普通文本不动它', () => {
    const t = '我的退款三天了还没到账';
    expect(mask(t)).toBe(t);
  });
});
