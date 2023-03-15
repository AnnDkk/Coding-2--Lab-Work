/*
 *  boid.cpp
 *  boids
 *
 *  Created by Marco Gillies on 05/10/2010.
 *  Copyright 2010 Goldsmiths, University of London. All rights reserved.
 *
 */

#include "boidnew.h"
#include "ofMain.h"

Boidnew::Boidnew()
{
	separationWeight = 1.0f;
	cohesionWeight = 0.2f;
	alignmentWeight = 0.1f;

	separationThreshold = 15;
	neighbourhoodSize = 100;

	position = ofVec3f(ofRandom(0, 200), ofRandom(0, 200));
	velocity = ofVec3f(ofRandom(-2, 2), ofRandom(-2, 2));
}
////
////Boidnew::Boidnew(ofVec3f &pos, ofVec3f &vel)
////{
////	separationWeight = 1.0f;
////	cohesionWeight = 0.2f;
////	alignmentWeight = 0.1f;
////
////	separationThreshold = 15;
////	neighbourhoodSize = 150;
////
////	position = pos;
////	velocity = vel;
////}
////
//Boidnew::~Boidnew()
////{
////
////}
////
////float Boidnew::getSeparationWeight()
////{
////	return separationWeight;
////}
////float Boidnew::getCohesionWeight()
////{
////	return cohesionWeight;
////}
////
////float Boidnew::getAlignmentWeight()
////{
////	return alignmentWeight;
////}
////
////
////float Boidnew::getSeparationThreshold()
////{
////	return separationThreshold;
////}
////
////float Boidnew::getNeighbourhoodSize()
////{
////	return neighbourhoodSize;
////}
////
////
////void Boidnew::setSeparationWeight(float f)
////{
////	separationWeight = f;
////}
////void Boidnew::setCohesionWeight(float f)
////{
////	cohesionWeight = f;
////}
////
////void Boidnew::setAlignmentWeight(float f)
////{
////	alignmentWeight = f;
////}
////
////
////void Boidnew::setSeparationThreshold(float f)
////{
////	separationThreshold = f;
////}
////
////void Boidnew::setNeighbourhoodSize(float f)
////{
////	neighbourhoodSize = f;
////}
////
////
////ofVec3f Boidnew::getPosition()
////{
////	return position;
////}
////
////ofVec3f Boidnew::getVelocity()
////{
////	return velocity;
////}
////
////ofVec3f Boidnew::separation(std::vector<Boidnew *> &otherBoids)
////{
////    ofVec3f v;
////	// finds the first collision and avoids that
////	// should probably find the nearest one
////	// can you figure out how to do that?
////	for (int i = 0; i < otherBoids.size(); i++)
////	{
////
////		if(position.distance(otherBoids[i]->getPosition()) < separationThreshold)
////		{
////            v = position -  otherBoids[i]->getPosition();
////			v.normalize();
////            return v;
////		}
////	}
////    return v;
////}
////
////ofVec3f Boid::cohesion(std::vector<Boid *> &otherBoids)
////{
////	ofVec3f average(0,0,0);
////	int count = 0;
////	for (int i = 0; i < otherBoids.size(); i++)
////	{
////		if (position.distance(otherBoids[i]->getPosition()) < neighbourhoodSize)
////		{
////			average += otherBoids[i]->getPosition();
////			count += 1;
////		}
////	}
////	average /= count;
////	ofVec3f v =  average - position;
////	v.normalize();
////	return v;
////}
////
////ofVec3f Boid::alignment(std::vector<Boid *> &otherBoids)
////{
////	ofVec3f average(0,0,0);
////	int count = 0;
////	for (int i = 0; i < otherBoids.size(); i++)
////	{
////		if (position.distance(otherBoids[i]->getPosition()) < neighbourhoodSize)
////		{
////			average += otherBoids[i]->getVelocity();
////			count += 5;
////		}
////	}
////	average /= count;
////	ofVec3f v =  average - velocity;
////	v.normalize();
////	return v;
////}
////
////void Boid::update(std::vector<Boid *> &otherBoids, ofVec3f &min, ofVec3f &max)
////{
////	velocity += separationWeight*separation(otherBoids);
////	velocity += cohesionWeight*cohesion(otherBoids);
////	velocity += alignmentWeight*alignment(otherBoids);
////
////	walls(min, max);
////	position += velocity;
////}
////
////void Boid::walls(ofVec3f &min, ofVec3f &max)
////{
////	if (position.x < min.x){
////		position.x = min.x;
////		velocity.x *= -1;
////	} else if (position.x > max.x){
////		position.x = max.x;
////		velocity.x *= -1;
////	}
////
////	if (position.y < min.y){
////		position.y = min.y;
////		velocity.y *= -1;
////	} else if (position.y > max.y){
////		position.y = max.y;
////		velocity.y *= -1;
////	}
////
////


void Boidnew::draw()
{
	ofSetColor(ofRandom(0,255), 88, 200);
	ofRect(position.x, position.y, 20, 20);
}
