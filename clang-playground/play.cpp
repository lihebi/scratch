#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Frontend/FrontendActions.h"
#include "llvm/Support/CommandLine.h"
#include "clang/Tooling/CommonOptionsParser.h"


using namespace clang;
using namespace clang::tooling;
using namespace llvm;

static cl::OptionCategory MyToolCategory("my-tool options");


/**
 * Extract code
 * with meta data
 * - begin
 * - end
 * - type
 * - name
 */
class MyVisitor
  : public RecursiveASTVisitor<MyVisitor> {
public:
  explicit MyVisitor(ASTContext *Context)
    : Context(Context) {}
  bool VisitCXXRecordDecl(CXXRecordDecl *Declaration) {
    return true;
  }

  // function
  // enum
  // structure
  // typedef
  // union
  // var
  bool VisitFunctionDecl(FunctionDecl *func_decl) {
    std::string name = func_decl->getNameInfo().getName().getAsString();
    errs() << "Function: " << name << "\n";
    SourceRange sourceRange = func_decl->getSourceRange();
    SourceLocation beginLoc = sourceRange.getBegin();
    SourceLocation endLoc = sourceRange.getEnd();
    FullSourceLoc fullBeginLoc = Context->getFullLoc(beginLoc);
    FullSourceLoc fullEndLoc = Context->getFullLoc(endLoc);
    unsigned begin_line = fullBeginLoc.getSpellingLineNumber();
    unsigned begin_col = fullBeginLoc.getSpellingColumnNumber();
    unsigned end_line = fullEndLoc.getSpellingLineNumber();
    unsigned end_col = fullEndLoc.getSpellingColumnNumber();
    
    return true;
  }

  std::pair<int,int> convertLocation(clang::SourceLocation loc) {
    clang::FullSourceLoc fullBeginLoc = Context->getFullLoc(loc);
    int line = fullBeginLoc.getSpellingLineNumber();
    int column = fullBeginLoc.getSpellingColumnNumber();
    return {line, column};
  }

  bool VisitEnumDecl(EnumDecl *decl) {
    errs() << decl->getName() << " ";
    for (auto it=decl->enumerator_begin();it!=decl->enumerator_end();++it) {
      errs() << (*it)->getName() << " ";
    }
    return true;
  }

  bool VisitVarDecl(VarDecl *decl) {
    errs() << decl->getName() << " ";
    // decl->dump();
    if (decl->hasExternalStorage()) {
      errs() << "extern ";
    }
    if (decl->hasInit()) {
      errs() << "but has init";
    }
    // defintion of var seems to be the assignment
    if (decl->getDefinition() == decl) {
      errs() << "ewqual";
    } else {
      errs() << "non-equal";
    }
    errs() << "\n";
    return true;
    // errs() << decl->getName() << " ";
  }

  bool VisitRecordDecl(RecordDecl *decl) {
    if (decl->isThisDeclarationADefinition()) {
      errs() << ".....";
      errs() << decl->getName();
      // if (decl->isAnonymousStructOrUnion()) {
      if (decl->getName().empty()) {
        errs() << " Is anonymous";
      }
      errs() << "\n";
    }
    return true;
  }
  
  bool VisitTypedefDecl(TypedefDecl *decl) {
    std::string name = decl->getName();
    errs() << " ";
    clang::SourceLocation begin = decl->getLocStart();
    clang::SourceLocation end = decl->getLocEnd();
    convertLocation(begin);
    convertLocation(end);

    NamedDecl *under_decl = decl->getUnderlyingDecl();

    QualType underlying = decl->getUnderlyingType();
    std::string underlying_str = underlying.getAsString();
    if (!underlying.isNull()) {
      const Type *type = underlying.getTypePtr();
      if (type) {
        const RecordType *record = nullptr;
        if (type->isStructureType()) {
          record = type->getAsStructureType();
        } else if (type->isUnionType()) {
          record = type->getAsUnionType();
        }
        if (record) {
          const RecordDecl *record_decl = record->getDecl();
          if (record_decl) {
            errs() << underlying_str;
            if (record_decl->getName().empty()) {
              errs() << "anonymous, skipped." << " ";
            } else {
              errs() << " => typedef " << underlying_str << " " << name;
            }
          }
        } else {
            errs() << "not record";
            errs() << " " << underlying_str;
          }
      }
    }
    errs() << "\n";
    /**
     * In order to declare a typedef in the top, i need:
     * 1. find if it is a pure typedef or a mix
     * 2. if it is a pure:
     *    - just put it forward
     *    if it is not:
     *    - get the body definition, and get the

     it can be:
     - define a struct :: extract decl of it
     - define a pure typedef :: put it forward
     - declare a struct :: NOTHING
     - define a typedef and a struct :: extract 1. pure typedef 2. struct
     - define a struct, and multiple typedefs, maybe with pointer :: extract 1. pure typedef, 2. struct
     */

    
    
    return true;
  }
private:
  ASTContext *Context;
};

class MyConsumer : public clang::ASTConsumer {
public:
  explicit MyConsumer(ASTContext *Context)
    : Visitor(Context) {}
  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
private:
  MyVisitor Visitor;
};



class MyAction : public clang::ASTFrontendAction {
public:
  virtual std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler, llvm::StringRef InFile) {
    return std::unique_ptr<clang::ASTConsumer>
      (new MyConsumer(&Compiler.getASTContext()));
  }
};


int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());
  Tool.run(newFrontendActionFactory<MyAction>().get());
}
