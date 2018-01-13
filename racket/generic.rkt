#lang racket

(require racket/generic)

(define-generics printable
  (gen-print printable port)
  (gen-port-print port printable)
  (more-print printable)
  (gen-print* printable [port] #:width width #:height height))

(define (foo) #f)

(struct mystruct (b [a #:mutable] c) #:prefab)
(struct child mystruct (d e) #:prefab)
(mystruct 3 4 5)
(define tmp (child 3 4 5 1 2))
(set-mystruct-a! tmp 8)
(child-a tmp)

(struct-copy mystruct tmp [b 3])

(struct num (v)
  #:prefab
  #:methods gen:printable
  [(define/generic alias gen-print)
   (define/generic alias2 gen-print*)
   ;; (define alias3 gen-print)
   (define (gen-print n port)
     (fprintf port "Num: ~a" (num-v n)))
   (define (gen-port-print port n)
     (let ([alias2 gen-print]) 
       (gen-print n port)
       (alias n port)
       ;; (alias2 n)
       ;; (alias3 n port)
       )
     )]
  )

(struct bool (v)
  #:methods gen:printable
  [(define (gen-print n port)
     (fprintf port "Bool: ~a" (bool-v n)))])

;; (gen-port-print (current-output-port) (num 8) )
;; (gen-print (bool 8))
