package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

// Config 全局配置,对应 Java 的 application.yml
type Config struct {
	Server struct {
		Port int `yaml:"port"`
	} `yaml:"server"`
	MySQL struct {
		DSN string `yaml:"dsn"`
	} `yaml:"mysql"`
	Redis struct {
		Addr     string `yaml:"addr"`
		Password string `yaml:"password"`
		DB       int    `yaml:"db"`
	} `yaml:"redis"`
	AI struct {
		Deepseek struct {
			APIKey  string `yaml:"apiKey"`
			BaseURL string `yaml:"baseUrl"`
			Model   string `yaml:"model"`
		} `yaml:"deepseek"`
		Ollama struct {
			BaseURL       string `yaml:"baseUrl"`
			EmbeddingModel string `yaml:"embeddingModel"`
		} `yaml:"ollama"`
	} `yaml:"ai"`
}

var cfg *Config

// Load 加载配置文件。
// 优先读 config.local.yaml(本地真实配置,gitignore 不进仓库),没有再读 config.yaml(占位)。
// config.yaml 里的 ${ENV} 占位用环境变量覆盖。
func Load(path string) (*Config, error) {
	// 优先本地配置(含真实 key)
	localPath := "config.local.yaml"
	if _, err := os.Stat(localPath); err == nil {
		path = localPath
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	c := &Config{}
	if err := yaml.Unmarshal(data, c); err != nil {
		return nil, err
	}
	// 环境变量覆盖 DeepSeek key(占位符 ${DEEPSEEK_API_KEY} 的兜底)
	if env := os.Getenv("DEEPSEEK_API_KEY"); env != "" {
		c.AI.Deepseek.APIKey = env
	}
	cfg = c
	return c, nil
}

func Get() *Config { return cfg }
