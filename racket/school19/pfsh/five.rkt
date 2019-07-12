#lang racket

;; (require racket/parse/define)

(require (for-syntax racket/base
                     syntax/parse
                     racket/list)
         syntax/parse/define
         racket/contract
         racket/list
         syntax/parse
         syntax/macro-testing
         rackunit)

(provide (rename-out [define-five-3 define]))

(define-simple-macro (define-five-1 name:id e:expr)
  (begin
    (displayln (format "Assuming ~a produces 5" 'e))
    (define name 5)))

(define-syntax (define-five-2 stx)
  (syntax-parse stx
    [(_ name:id e:expr)
     #'(begin
         (displayln (format "Assuming ~a produces 5" 'e))
         (define name 5))]))

(define-syntax (define-five-3 stx)
  (define es (syntax->list stx))
  (define name (list-ref es 1))
  (define e (list-ref es 2))
  (displayln (format "Assuming ~a produces 5"
                     (syntax->datum e)))
  #`(define #,name #,e)
  #`(define #,name 5))
