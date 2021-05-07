package edu.coursera.concurrent;

import edu.rice.pcdp.Actor;

import static edu.rice.pcdp.PCDP.finish;

/**
 * An actor-based implementation of the Sieve of Eratosthenes.
 * <p>
 * TODO Fill in the empty SieveActorActor actor class below and use it from
 * countPrimes to determin the number of primes <= limit.
 */
public final class SieveActor extends Sieve {
    /**
     * {@inheritDoc}
     * <p>
     * TODO Use the SieveActorActor class to calculate the number of primes <=
     * limit in parallel. You might consider how you can model the Sieve of
     * Eratosthenes as a pipeline of actors, each corresponding to a single
     * prime number.
     */
    @Override
    public int countPrimes(final int limit) {
        SieveActorActor a = new SieveActorActor(2);
        SieveActorActor finalA = a;
        finish(() -> {
            for (int i = 3; i <= limit; i++) {
                finalA.ComputeNum(i);
            }
        });

        int totalCount = 0;
        while (a != null) {
            totalCount += a.GetPrimeNumber();
            a = a.nextActor;
        }
        return totalCount;
    }

    /**
     * An actor class that helps implement the Sieve of Eratosthenes in
     * parallel.
     */
    public enum MessageType {
        eNewPrime,
        eNumber,
    }

    public static final class Message {
        MessageType type;
        int num;

        public Message(MessageType type, int num) {
            this.type = type;
            this.num = num;
        }
    }

    public static final int NUMOFMAXPRIMES = 4;

    public static final class SieveActorActor extends Actor {

        public SieveActorActor(int number) {
            this.primes = new int[NUMOFMAXPRIMES];
            this.primes[0] = number;
            this.primeNumber = 1;
            this.nextActor = null;
        }

        public int GetPrimeNumber() {
            return primeNumber;
        }


        public void ComputeNum(int num) {
            this.send(new Message(MessageType.eNumber, num));
        }

        public SieveActorActor nextActor;
        public int totalCount;
        public int[] primes;
        public int primeNumber = 0;

        @Override
        public void process(final Object msg) {
            Message m = (Message) msg;
            switch (m.type) {
                case eNewPrime:
                    totalCount = totalCount + 1;
                    break;
                case eNumber:
                    int num = m.num;
                    for (int i = 0; i < this.primeNumber; i++) {
                        if (num % this.primes[i] == 0)
                            return;
                    }

                    if (this.primeNumber < NUMOFMAXPRIMES) {
                        this.primes[primeNumber] = num;
                        this.primeNumber++;
                        break;
                    } else if (this.nextActor != null) {
                        this.nextActor.send(msg);
                        break;
                    } else {
                        SieveActorActor a = new SieveActorActor(num);
                        this.nextActor = a;
                    }

            }
        }
    }
}

