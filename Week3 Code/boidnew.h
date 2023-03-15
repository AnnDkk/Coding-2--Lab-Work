/*

 *
 */

#ifndef _BOIDnew
#define _BOIDnew
#include <vector>
#include "ofMain.h"
#include "boid.h"

class Boidnew : public Boid
{
// all the methods and variables after the
// private keyword can only be used inside
// the class
//protected:
//	ofVec3f position;
//	ofVec3f velocity;
//
//	float separationWeight;
//	float cohesionWeight;
//	float alignmentWeight;
//
//	float separationThreshold;
//	float neighbourhoodSize;
//
//	ofVec3f separation(std::vector<Boidnew *> &otherBoids);
//	ofVec3f cohesion(std::vector<Boidnew *> &otherBoids);
//	ofVec3f alignment(std::vector<Boidnew *> &otherBoids);
//
// all the methods and variables after the
// public keyword can only be used by anyone
public:
	Boidnew();
//	Boidnew(ofVec3f &pos, ofVec3f &vel);
//
//	~Boidnew();
//
//	ofVec3f getPosition();
//	ofVec3f getVelocity();
////
//
//	float getSeparationWeight();
//	float getCohesionWeight();
//	float getAlignmentWeight();
//
//	float getSeparationThreshold();
//	float getNeighbourhoodSize();
//
//	void setSeparationWeight(float f);
//	void setCohesionWeight(float f);
//	void setAlignmentWeight(float f);
//
//	void setSeparationThreshold(float f);
//	void setNeighbourhoodSize(float f);
//
//	void update(std::vector<Boidnew *> &otherBoids, ofVec3f &min, ofVec3f &max);
//
//	void walls(ofVec3f &min, ofVec3f &max);
//
	void draw();
};

#endif
