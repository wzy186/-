package utils

// Result 统一返回结构,对齐 Java 的 com.dss.dto.Result
type Result struct {
	Success bool        `json:"success"`
	Code    int         `json:"code"`
	Msg     string      `json:"msg"`
	Data    interface{} `json:"data"`
}

func OK() *Result { return &Result{Success: true, Code: 200, Msg: "success"} }

func OKWith(data interface{}) *Result { return &Result{Success: true, Code: 200, Msg: "success", Data: data} }

func Fail(msg string) *Result { return &Result{Success: false, Code: 500, Msg: msg} }

func FailWith(code int, msg string) *Result { return &Result{Success: false, Code: code, Msg: msg} }
