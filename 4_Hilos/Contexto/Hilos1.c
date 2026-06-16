#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<unistd.h>

void *Hilo1(void *argumentos)
{
   int i;
   for(i=0;i<8;i++)
   {
      printf("\nDentro del hilo...\n");
      sleep(1);
   }
   pthread_exit(NULL);
}

int main ()
{
   pthread_t id_hilo1;

   printf("\nCreacion del hilo...\n");
   
   pthread_create(&id_hilo1,NULL,Hilo1,NULL);
   
   sleep(3);
   
   printf("\nHilo creado. Esperando su finalizacion...\n");
   
   pthread_join(id_hilo1,NULL);
   
   printf("\nHilo finalizado...\n");
   return 0;
}
