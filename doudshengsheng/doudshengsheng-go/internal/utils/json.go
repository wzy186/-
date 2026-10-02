package utils

import "encoding/json"

// ToJSON 序列化
func ToJSON(v interface{}) []byte {
	b, _ := json.Marshal(v)
	return b
}

// UnmarshalJSON 反序列化
func UnmarshalJSON(data []byte, v interface{}) error {
	return json.Unmarshal(data, v)
}
