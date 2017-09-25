;; define function
(define square)

;; control structure
(if #t 1 2)
(if #f 1 2)
(if (> 1 2)
    (+ 1 2)
    (- 1 2))

;; pair & list
(car (list 1))

(null? (list))

(list 1 2 3 4 5)
'(1 2 3 4 5)
(list 1 2 "hello" 3)
(car (list 1 2 3))
(cdr (list 1 2 3))
(cdr (cdr (list 1 2 3)))
(cdr (list 1))

(cons 1 2)
(cons 3 (list 3))
(cons 1 '())
(cons 1 #t)


(define mylist '((1 2) (3 4)))
(cons '(5 6) mylist)
(car (car (cdr mylist)))


(define my-cadr)
(define my-caddr)

(define my-last)
(define my-append)
(define my-length)
(define my-max)
(define my-reverse)

;; high order function
(define arbitrary-arith
  (lambda (op a b)
    (op a b)))

(define add
  (lambda (a b)
    (+ a b)))
(define minus
  (lambda (a b)
    (- a b)))

(arbitrary-arith add 1 2)
(arbitrary-arith + 1 2)

;; data structure

(define pair
  (lambda (fst snd)
    (lambda (op)
      (if op fst snd))))

(pair 1 2)

((pair 1 2) #t)
((pair 1 2) #f)

;; curry
(define curry-add
  (lambda (x)
    (lambda (y)
      (+ x y))))

((curry-add 1) 2)

