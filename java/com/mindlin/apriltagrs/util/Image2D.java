package com.mindlin.apriltagrs.util;

import java.nio.ByteBuffer;
import java.util.Objects;

public interface Image2D {
    static void validate(Image2D image) throws NullPointerException, IllegalArgumentException {
        Objects.requireNonNull(image);
        var buffer = Objects.requireNonNull(image.buffer());
        int expectedCapacity = validateDimensions(image.getWidth(), image.getHeight(), image.getStride());
        if (expectedCapacity != buffer.remaining())
            throw new IllegalArgumentException("Capacity mismatch");
    }
    private static int validateDimensions(int width, int height, int stride) throws IllegalArgumentException{
        if (height <= 0)
            throw new IllegalArgumentException("Non-positive height");
        if (width <= 0)
            throw new IllegalArgumentException("Non-positive width");
        if (stride < width)
            throw new IllegalArgumentException("Stride is smaller than width");

        try {
            return Math.multiplyExact(height, width);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("Capacity too large", e);
        }
    }
    static Image2D from(ByteBuffer buffer, int width, int height) {
        return Image2D.from(buffer, width, height, width);
    }
    static Image2D from(ByteBuffer buffer, int width, int height, int stride) {
        return new WrappedBuffer(buffer, width, height, stride);
    }
    static Image2D from(byte[][] image) {
        Objects.requireNonNull(image);
        int height = image.length;
        if (height == 0)
            throw new IllegalArgumentException("Non-positive height");
        int width = Objects.requireNonNull(image[0], "Row 0").length;
        int expectedCapacity = validateDimensions(width, height, width);

        var buffer = ByteBuffer.allocateDirect(expectedCapacity);
        for (int i = 0; i < height; i++) {
            var rowId = i;
            var row = Objects.requireNonNull(image[i], () -> "Row " + rowId);
            if (row.length != width)
                throw new IllegalArgumentException(String.format("Row %d length mismatch (actual: %d; expected: %d)", rowId, row.length, width));
            buffer.put(row);
        }
        return from(buffer, width, height);
    }

    int getWidth();
    int getHeight();
    int getStride();
    ByteBuffer buffer();

    final class WrappedBuffer implements Image2D {
        private final ByteBuffer buffer;
        private final int width;
        private final int height;
        private final int stride;
        WrappedBuffer(ByteBuffer buffer, int width, int height, int stride) {
            this.buffer = Objects.requireNonNull(buffer);
            this.width = width;
            this.height = height;
            this.stride = stride;
            Image2D.validate(this);
        }

        @Override
        public int getWidth() {
            return this.width;
        }

        @Override
        public int getHeight() {
            return this.height;
        }

        @Override
        public int getStride() {
            return this.stride;
        }

        @Override
        public ByteBuffer buffer() {
            return this.buffer.duplicate();
        }
    }
}
