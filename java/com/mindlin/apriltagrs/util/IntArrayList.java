package com.mindlin.apriltagrs.util;

import java.util.Arrays;
import java.util.Objects;

public class IntArrayList {
    int[] data;
    int size;

    public IntArrayList(int capacity) {
        this.data = new int[capacity];
        this.size = 0;
    }

    public IntArrayList(int[] data) {
        this.data = Objects.requireNonNull(data);
        this.size = data.length;
    }

    public int size() {
        return this.size;
    }

    public void push(int value) {
        if (this.size == this.data.length) {
            int newSize = Math.max(this.data.length * 2, 16);
            if (newSize < this.data.length)
                newSize = Integer.MAX_VALUE;
            this.data = Arrays.copyOf(this.data, newSize);
        }
        this.data[this.size++] = value;
    }

    public int[] getData() {
        return this.data;
    }

    public int[] toIntArray() {
        if (this.data.length == this.size) {
            return this.data;
        } else {
            return Arrays.copyOf(this.data, this.size);
        }
    }
}
