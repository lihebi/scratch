;; define function
(define square
  (lambda (x) (* x x)))
(square 3)

;; control structure
(if #t 1 2)
(if #f 1 2)
(if (> 1 2)
    (+ 1 2)
    (- 1 2))

;; pair & list

(cons 1 2)
(cons 2 (cons 1 (cons 2 '())))
(list 1 2 3)
'(1 2 3)
(car (cons 1 2))
(cdr (cons 1 2))

(car (list 1 2 3))
(car (cdr (cdr (list 1 2 3))))

(cons 1 2)
(cons 1 '(2 3))
'(12 "fna" 3)

(define mylist '((1 2) (3 4)))
(cons '(5 6) mylist)
(car (car (cdr mylist)))


(define my-cadr
  (lambda (lst)
    (car (cdr lst))))
(define my-caddr)

(define my-last
  (lambda (lst)
    (if (= 1(length lst))
        (car lst)
        (my-last (cdr lst)))))
(my-last '(1 2 3))

(append '(1 2 3) '(4 5 6) '(3 4))
(define my-append
  (lambda (a b)
    (if (null? a)
        b
        (cons (car a)
              (my-append (cdr a) b)))))
(my-append '(1 2) '(3 4))
(define my-length
  (lambda (lst)
    (if (null? lst)
        0
        (+ 1 (my-length (cdr lst))))))
(my-length '(1 2 3))
(max 1 2 3)
(my-max '(-1 -3 -2 -1))
(define my-max
  (lambda (lst)
    (if (null? lst)
        0
        (if (> (car lst)
               (my-max (cdr lst)))
            (car lst)
            (my-max (cdr lst))))))
(reverse '(1 2 3))
(define my-reverse
  (lambda (lst)
    (if (null? lst)
        '()
        (append (my-reverse (cdr lst))
                (list (car lst)))
        )))
(my-reverse '(1 2 3))
(define my-reverse-2
  (lambda (lst res)
    (if (null? lst)
        res
        (my-reverse-2
         (cdr lst)
         (cons (car lst) res)))))
(define my-wrapper
  (lambda (lst)
    (my-reverse-2 lst '())))
(my-wrapper '(2 3 4))
(my-reverse-2 '(2 3 4) '())

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

(list (list 1 2) '(3 4))

(list (cons 1 2) (cons 3 4))

(pair 1 2)

((pair 1 2) #t)
((pair 1 2) #f)

;; curry
(define curry-add
  (lambda (x)
    (lambda (y)
      (+ x y))))

((curry-add 1) 2)

