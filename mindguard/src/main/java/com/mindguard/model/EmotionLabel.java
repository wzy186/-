package com.mindguard.model;

public enum EmotionLabel {
    NORMAL("正常"),
    ANXIOUS("焦虑"),
    DEPRESSED("抑郁"),
    ANGRY("愤怒"),
    FEARFUL("恐惧"),
    HOPELESS("绝望"),
    SUICIDAL("自杀倾向");

    private final String label;

    EmotionLabel(String label) {
        this.label = label;
    }

    public String getLabel() {
        return label;
    }
}
