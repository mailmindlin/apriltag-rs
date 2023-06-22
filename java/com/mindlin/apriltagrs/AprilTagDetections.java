package com.mindlin.apriltagrs;

import java.util.AbstractList;
import java.util.Collection;
import java.util.Optional;
import java.util.RandomAccess;
import com.mindlin.apriltagrs.debug.TimeProfile;

public final class AprilTagDetections extends AbstractList<AprilTagDetection> implements RandomAccess {
    int numQuads;
    TimeProfile tp;

    public int getNumQuads() {
        return this.numQuads;
    }

    public Optional<TimeProfile> getTimeProfile() {
        return Optional.ofNullable(this.tp);
    }
    
    @Override
    public int size() {
        
    }

    @Override
    public AprilTagDetection get(int index) {
        
    }

    @Override
    public boolean addAll(Collection<? extends AprilTagDetection> c) {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public boolean addAll(int index, Collection<? extends AprilTagDetection> c) {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public void clear() {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }
}