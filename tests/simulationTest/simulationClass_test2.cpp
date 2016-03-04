#include <iostream>
#include <simulation/types/vector.hpp>//Vector
#include <simulation/simulation.hpp> //Simulation
#include <cstdlib>
#include <ctime>

#define N_BODIES 100
#define SMOOTHNESS 2
#define GRAV 0.1f
#define DTIME 0.01f
#define STEPS 1000
#define INNER_STEP 10

using namespace nbody::simulation;

int main (){
        //Init rand
        srand(static_cast<unsigned>(time(0)));
        
        //Init Bodies
        types::Vector<4,float>bodiesPosition[N_BODIES];
        types::Vector<3,float>bodiesVelocity[N_BODIES];
        for(int i = 0; i< N_BODIES;i++)
        {
            bodiesPosition[i]={
                static_cast<float>( rand() % 101 ) - 50.0f,
                static_cast<float>( rand() % 101 ) - 50.0f,
                static_cast<float>( rand() % 101 ) - 50.0f,
                static_cast<float>( (rand())*2/RAND_MAX +2.0f)
                        
            };
            bodiesVelocity[i] = {
               ( static_cast<float>( rand() % 3 ) - 1.0f ) / 10.f,
               ( static_cast<float>( rand() % 3 ) - 1.0f ) / 10.f,
               ( static_cast<float>( rand() % 3 ) - 1.0f ) / 10.f
            };
        }
        //Init Simulation
        Simulation<
            4,
            float,
            float,
            std::size_t>sim(
                bodiesPosition,
                bodiesVelocity,
                N_BODIES,
                SMOOTHNESS,
                GRAV);

        //Print masses
        for (unsigned int i(0);i<N_BODIES;i++)
        {
            std::cout<<bodiesPosition[i][3];
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
            types::Vector<4,float> * positions =sim.getPositions();
            for (unsigned int i(0); i<N_BODIES; i++)
            {
                types::Vector<4,float> position= positions[i];
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

