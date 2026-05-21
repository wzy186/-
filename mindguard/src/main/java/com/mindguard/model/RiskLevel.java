package com.mindguard.model;

public enum RiskLevel {
    LOW("低风险"),
    MEDIUM("中风险"),
    HIGH("高风险"),
    CRITICAL("极高风险");

    private final String label;

    RiskLevel(String label) {
        this.label = label;
    }

    public String getLabel() {
        return label;
    }
}
