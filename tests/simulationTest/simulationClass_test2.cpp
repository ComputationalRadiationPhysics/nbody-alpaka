#include <iostream>
#include <simulation/types/vector.hpp>//Vector
#include <simulation/simulation.hpp> //Simulation
#include <cstdlib>
#include <ctime>

#define N_BODIES 20
#define SMOOTHNESS 1.1755e-38
#define GRAV 6.67408e-10
#define DTIME 36000
#define STEPS 100
#define INNER_STEP 1000

using namespace nbody::simulation;

int main (int argc, char *argv[]){
        //Init rand
        srand(static_cast<unsigned>(time(0)));
        
        //Init Bodies
        float bodiesMass[N_BODIES];
        types::Vector<3,float>bodiesPosition[N_BODIES];
        types::Vector<3,float>bodiesVelocity[N_BODIES];
        for(int i = 0; i< N_BODIES;i++)
        {
            bodiesMass[i]=2e30 *(0.5f+static_cast<float>(rand()/static_cast<float>(RAND_MAX/(10.0f-0.5f))));
            bodiesPosition[i]={static_cast<float>(rand()/(static_cast<float> (RAND_MAX/3))),
            static_cast<float>(rand()/(static_cast<float> (RAND_MAX/3))),
            static_cast<float>(rand()/(static_cast<float> (RAND_MAX/3)))
            };
            bodiesPosition [i]= 9.5e12*bodiesPosition[i];
            bodiesVelocity[i] = {0,0,0};
        }
        //Init Simulation
        Simulation<
            3,
            float,
            float,
            std::size_t>sim(
                bodiesPosition,
                bodiesVelocity,
                bodiesMass,
                N_BODIES,
                SMOOTHNESS,
                GRAV);
        for (unsigned int i(0); i<100;i++) sim.step(DTIME); 

        //Print masses
        for (unsigned int i(0);i<N_BODIES;i++)
        {
            std::cout<<bodiesMass[i];
            if(i!=N_BODIES-1) std::cout <<" ";
        }
        std::cout<<std::endl;
		std::cerr<<"[";
		for(unsigned int i(0); i<STEPS;i++)
		{
			std::cerr<<".";
		}
		std::cerr<<"]"<<std::endl<<" ";
        //RunSimulation
        for(unsigned int s(0); s<STEPS;s++)
        {	
			std::cerr<<".";
		
            //Print positions
            types::Vector<3,float> * positions =sim.getPositions();
            for (unsigned int i(0); i<N_BODIES; i++)
            {
                types::Vector<3,float> position= positions[i];
                std::cout<<position[0]
                    <<";"
                    <<position[1]
                    <<";"
                    <<position[2];
                if (i!=N_BODIES-1)
                    std::cout<<"|";
            }
            std::cout<<std::endl;
            //innersteps
            for (unsigned int j(0);j<INNER_STEP;j++)
                sim.step(DTIME);
        }
	return 0;
}

