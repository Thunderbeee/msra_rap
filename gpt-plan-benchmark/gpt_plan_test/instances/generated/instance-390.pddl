(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g b a h c f e l d)
(:init 
(handempty)
(ontable g)
(ontable b)
(ontable a)
(ontable h)
(ontable c)
(ontable f)
(ontable e)
(ontable l)
(ontable d)
(clear g)
(clear b)
(clear a)
(clear h)
(clear c)
(clear f)
(clear e)
(clear l)
(clear d)
)
(:goal
(and
(on g b)
(on b a)
(on a h)
(on h c)
(on c f)
(on f e)
(on e l)
(on l d)
)))