#lang racket/base

;; (symbol? '5)

(require (for-syntax racket/base
                     syntax/parse)
         syntax/parse/define
         racket/contract
         racket/list
         syntax/parse)

#;
(begin-for-syntax
  (define-syntax-class))


;; (define (bigger-string x)
;;   (number->string (add1 x)))
;; (provide
;;  (contract-out
;;   [bigger-string
;;    (-> number? string?)]))

(begin-for-syntax
  #;
  (define-splicing-syntax-class predicate
    (pattern (~)))
  (define-syntax-class contract-class
    (pattern ((~datum ->) p:identifier ...))))

(define-syntax (define/contract-out stx)
  (syntax-parse stx
    [(_ (func-name arg ...) func-body contract)
     #:declare func-name identifier
     #:declare arg identifier
     #:declare contract contract-class
     #'(begin
         (define (func-name arg ...) func-body)
         (provide
          (contract-out
           [func-name
            contract]))
         )]))

(convert-syntax-error)

(define/contract-out
  (bigger-string x)
  ;; (5 x)
  (number->string (add1 x))
  ;; (-> number? string?)
  (-> (-> number? string?) number?)
  )

(bigger-string 8)



;; (number->string (add1 x))
;; (fseq2 x add1 number->string)


(define-syntax (fseq2 stx)
  (syntax-parse stx
    [(_ arg foo bar)
     #'(bar (foo arg))]))

(fseq2 2 add1 number->string)


(define-syntax (fseq stx)
  (syntax-parse stx
    [(_ arg foo ...)
     ;; #'(bar (foo arg))
     ;; 1. reverse the list of functions
     ;; 2. apply recursively
     #'(foldr (λ (v res)
                (v res))
              arg
              (list foo ...))
     ]))

#;
(define-syntax (fseq stx)
  (syntax-parse stx
    [(_ arg)
     #'arg]
    [(_ arg foo more ...)
     #'(fseq (foo arg) more ...)]))

;; (apply add1 '(2))

;; (fseq 38 add1 number->string string-length) 


(define-syntax (simple-for/list3 stx)
  (syntax-parse stx
    ; 1. We will have more input pattern cases.
    [(_ ([elem-name seq] ...) computation)
     ; 2. We will do more than just return the output template.
     #'(letrec ([iterate
                 (λ (elem-name ...)
                   (cond
                     [(or (empty? elem-name) ...)
                      empty]
                     [else
                      (cons
                       (let ([elem-name (first elem-name)] ...)
                         computation)
                       (iterate (rest elem-name) ...))]))])
         (iterate seq ...))]))

(simple-for/list3 ([s (list "Rickard" "Brandon")]
                   [i (list 1 2 3)])
                  (string-append (number->string i) ". Burnt" s))

(define-syntax (simple-for/and stx)
  (syntax-parse stx
    [(_ ([elem-name seq] ...) computation)
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
         (iterate seq ...))]))

(simple-for/and ([x (list 2 4 6)]) (even? x))


#;
(define-syntax (simple-for/fold stx)
  (syntax-parse stx
    [(_ ([acc-name acc-value])
        ([elem-name seq] ...)
        computation)
     #'(letrec ([iterate
                 (λ (elem-name ...)
                   (cond
                     [(or (empty? elem-name) ...)
                      #t]
                     [else
                      (iterate
                       (let ([elem-name (first elem-name)] ...)
                         computation)
                       acc)
                      (let ([acc-name acc-value
                                      ])
                        (iterate (rest elem-name) ...))]))])
         (iterate seq ...))]))

#;
(simple-for/fold ([acc #t])
                 ([x (list 2 4 6)])
                 (and (even? x) acc))
