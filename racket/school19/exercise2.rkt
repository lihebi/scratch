#lang racket/base

;; (symbol? '5)

(require (for-syntax racket/base
                     syntax/parse
                     racket/list)
         syntax/parse/define
         racket/contract
         racket/list
         syntax/parse
         syntax/macro-testing
         rackunit)

(begin-for-syntax
  #;
  (define-splicing-syntax-class predicate
    (pattern (~)))
  (define-syntax-class contract-class
    (pattern ((~datum ->) p:identifier ...))))

;; (~literal)

(define-syntax (define/contract-out stx)
  (syntax-parse stx
    [(_ (func-name arg ...) func-body contract)
     #:declare func-name identifier
     #:declare arg id
     ;; #:declare contract contract-class
     #'(begin
         (define (func-name arg ...) func-body)
         (provide
          (contract-out
           [func-name
            contract]))
         )]))

;; (convert-syntax-error)

(check-exn #rx"^lambda: bad syntax$"
           (lambda () (convert-syntax-error
                       ;; (lambda (x) 3)
                       ;; (λ)
                       (lambda)
                       )))
(check-not-exn
 (lambda () (convert-syntax-error
             (lambda (x) 3)
             )))

(check-exn
 #rx"expected identifier"
 ;; (or/c (-> any/c any/c) regexp?)
 (λ ()
   (convert-syntax-error
    (define/contract-out
      ;; (bigger-string x)
      (5 x)
      (number->string (add1 x))
      (-> number? string?)
      ;; (-> (-> number? string?) number?)
      ))))

#;
(define/contract-out
  (bigger-string x)
  ;; (5 x)
  (number->string (add1 x))
  (-> number? string?)
  ;; (-> (-> number? string?) number?)
  )

;; (bigger-string 8)

(define-syntax (robust-for/list9 stx)
  (syntax-parse stx
    [(_ ([elem-name:id seq]) computation)
     #:declare seq (expr/c #'list?)
     #:with seq-w/-ctc (attribute seq.c)
     #'(map (λ (elem-name) computation) seq-w/-ctc)]))

(check-not-exn
 (λ () (robust-for/list9 ([x (list 1 2 3)]) (add1 x))))

(check-exn
 #rx"robust-for/list9: contract violation"
 (λ ()
   (convert-syntax-error
    (robust-for/list9 ([x 1]) (add1 x)))))


(begin-for-syntax
  (define-syntax-class binding1
    (pattern [x:id xe:expr])
    (pattern [xe:expr #:as x:id]))
  (define-syntax-class unique-bindings0
    (pattern (b:binding1 ...)
             #:fail-when
             (check-duplicates
              (syntax->list #'(b.x ...))
              free-identifier=?)
             "Duplicate binding")))


(begin-for-syntax
  (define-syntax-class my-formal
    (pattern x:id))
  (define-syntax-class my-formals
    (pattern (f:my-formal ...)
             #:with (x ...) #'(f.x ...)
             #:fail-when
             (check-duplicates
              (syntax->list #'(f.x ...))
              free-identifier=?)
             "Duplicate binding")))

(define-simple-macro (my-lambda fs:my-formals body:expr ...+)
  (lambda fs body ...))


(check-not-exn
 (λ () (convert-syntax-error
        (my-lambda (x) x))))
(check-not-exn
 (λ () (convert-syntax-error
        (my-lambda (x y) x))))
(check-exn
 #rx"my-lambda"
 (λ () (convert-syntax-error
        (my-lambda (3) x))))
(check-exn
 #rx"my-lambda"
 (λ () (convert-syntax-error
        (my-lambda (x x) x))))

(define-simple-macro (with-eight name body)
  (+ body (let ([name 8]) body)))
;; (format-id)

(let ([tmp 7])
  (+ tmp (with-eight tmp tmp)))




(define-syntax (simple-for/and stx)
  (syntax-parse stx
    [(_ ([elem-name seq] ...) computation)
     #:declare elem-name identifier
     #:declare seq (expr/c #'list)
     #'(letrec ([iterate
                 (λ (elem-name ...)
                   (cond
                     [(or (empty? elem-name) ...)
                      #t]
                     [else
                      (and
                       (let ([elem-name (first elem-name)] ...)
                         computation)
                       (iterate (rest elem-name) ...)

                       )]))])
         (iterate seq.c ...))]))

#;(simple-for/and ([x 2]) (even? x))

(check-exn
 #rx"simple-for/and"
 ;; #rx"contract violation"
 (λ ()
   (convert-syntax-error
    (simple-for/and ([x
                      ;; (list 2 4 6)
                      2
                      ]) (even? x)))))

(check-exn #rx"^lambda: bad syntax$"
           (lambda () (convert-syntax-error
                       ;; (lambda (x) 3)
                       ;; (λ)
                       (lambda)
                       )))
