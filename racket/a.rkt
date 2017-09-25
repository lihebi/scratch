;; functions
(define square
  (lambda (x)
    (* x x)))

(square 2)

;; control structure
(if #t 1 2)
(if #f 1 2)
(if (> 1 2)
    (+ 1 2)
    (- 1 2))

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

(define my-cadr
  (lambda (lst)
    (car (cdr lst))))

(define my-caddr
  (lambda (lst)
    (car (cdr (cdr lst)))))

(append '(1 2) '(3 4))
(define my-append
  (lambda (lst1 lst2)
    (if (null? lst1) lst2
        (if (null? lst2) lst1
            (cons (car lst1) (my-append (cdr lst1) lst2))))))

;; functions on list
(length '(1 2 3))
(define my-length
  (lambda (lst)
    (if (null? lst) 0
        (+ 1 (my-length (cdr lst))))))
(my-length '(1 2 3))

(max 1 2 3)
(define my-max
  (lambda (lst)
    (if (null? lst) 0
        (if (> (car lst) (my-max (cdr lst)))
            (car lst)
            (my-max (cdr lst))))))

(my-max '(1 2 3))
(my-max '(2 3 1))
(my-max '())
(my-max '(-1))

(reverse '(1 2 3))
(define my-reverse
  (lambda (lst)
    (if (null? lst) '()
        (append (my-reverse (cdr lst)) (list (car lst))))))
(my-reverse '(1 2 3))

;; high order functions
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
