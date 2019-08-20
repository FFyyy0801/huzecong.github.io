---
layout:     post
title:      "你所不知道的 template"
date:       2015-09-19 23:27:00 +0800
categories: tech
locale:     zh-Hans
toc:        true
---

> C makes it easy to shoot yourself in the foot; C++ makes it harder, but when you do it blows your whole leg off.
> 
> —— C++ 之父，Bjarne Stroustrup

C++ 是一门非常强大而复杂的语言，而**模板**（template）则是其“主打功能”之一。可以说，没有了模板，C++ 就不会是 C++，它不会有现在这样的灵活性和可拓展性，也不会像现在这样因为过于复杂而受人诟病。

本文将介绍 C++ 模板的一些较为少见但是非常强大的应用。为了读懂本文，你需要：

- 有基础的编程知识和经验；
- 能够看懂 C++ 代码；
- 对于未知的事物充满好奇心。

如果你具备上面这些条件，那么就做好准备认识你所不知道的 template 吧！

## 从基础开始

为了照顾不太明白的读者，我们先很简单地讲一下什么是 C++ 的模板。已经了然于心的读者们可以选择再来复习一遍，或者跳过这一小节。

在传统的 C 语言中，不同的函数不能有相同的名字。这个规定其实非常好理解，因为很多时候函数操作会依赖于特定的类型。但我们也没法排除有一些不依赖于类型的通用算法，如果要为每种类型都定义一个名字不同的函数，再在每个函数里实现一遍算法，岂不是很麻烦？

于是 C++ 标准委员会（ISO C++ committee）就想，我们要在 C++ 里加入一个新功能，使得程序员可以写出不依赖于具体类型的**泛型**（generic）代码。这个新功能就是模板。在 C++ 中，我们可以通过如下语法定义一个模板函数：

``` c++
template <class T>
T max(T a, T b) {
	if (a > b) return a;
	return b;
}
```

从函数名和函数的执行过程我们很容易推断出，它的作用就是找出并返回`a`和`b`中的较大值。但是这个`T`是个什么玩意儿呢？

其实`T`是在代码第一行里定义的`class T`，它代表的是一个*任意的类型*。也就是说，这个函数接受两个类型相同的东西，并返回一个同样类型的东西。

既然有了模板函数，我们也可以弄出一个模板类：

``` c++
template <int N, int M, typename T>
class Matrix {
private:
	T array[N][M];
public:
	Matrix();
	static Matrix identityMatrix();
	T &elementAtIndex(int x, int y) {
		return array[x][y];
	}
	const T &elementAtIndex(int x, int y) const {
		return array[x][y];
	}
} ;
```

注意到我们可以指定多个泛型的参数，而且这些参数还不一定得是`T`这样的*类型参数*——它还可以是*非类型参数*，如这里的整型`N`，甚至可以是嵌套的模板参数。

这些非常简单的例子可以让我们略微感受到模板的强大之处：只要类型`T`可以拷贝构造、定义了大于运算符，就可以套用这个函数。如果我们有心，可以用模板实现一整套泛型的算法，并提供简单的借口。设想一下，当我们要给自定义的类型排序的时候，不需要手写快排，而定义比较操作符，直接调用一个模板函数即可；当我们要使用某些数据结构的时候，直接把我们的类名告诉模板类即可。

事实上 C++ 的 STL 就是这样一个玩意儿。STL 的全称是 Standard Template Library，里面用模板实现了各种泛型的算法和数据结构。随叫随到，即写即用，从此无心造轮子，管他开不开 O2。

限于篇幅原因，模板的基础知识只能介绍到这。有关模板的更多知识，推荐大家阅读这个网页：[https://isocpp.org/wiki/faq/templates](https://isocpp.org/wiki/faq/templates)。

## 模板元编程

模板当然是个好东西，它非常之强大，可以完成原来写C代码时想都不敢想的功能。但问题也就在于它太强大了，就连当初设计模板的人也不知道它究竟有多么厉害。事实上，可以证明模板的这套语言本身就是图灵完备的，也就是说，光是使用模板，我们就可以在编译时完成一切计算。这就是所谓的**模板元编程**（template metaprogramming，TMP）。

### 从斐波那契谈起

如果要你用正常的 C++ 求斐波那契数列，你会怎么写？当然会是像下面这样：

``` c++
const int N = 100;
int a[N + 1];
a[0] = 0, a[1] = 1;
for (int i = 2; i <= N; ++i)
	a[i] = a[i - 1] + a[i - 2];
```

或者是直接写递归的版本：

``` c++
int fib(int n) {
	if (n == 0) return 0;
	if (n == 1) return 1;
	return fib(n - 1) + fib(n - 2);
}
```

其实我们可以用模板来做：

``` c++
template <int N>
struct Fib {
	static const int value = Fib<N-1>::value + Fib<N-2>::value;
} ;

template <>
struct Fib<0> {
	static const int value = 0;
} ;

template <>
struct Fib<1> {
	static const int value = 1;
} ;
```

我们在`Fib`类里定义了一个静态的常量`value`，代表第`N`个斐波那契数的值。`N = 0`和`N = 1`的两个模板特化是递归的边界条件，一般的情况则直接利用公式递归计算。整个计算过程都是在编译时完成的，而且由于模板的实例化机制，这个递归的过程还是记忆化的，即同一个斐波那契数不会被计算两次，计算的复杂度为 $O(N)$ 而非 $O(fibonacci(N))$ 。

如果我们要使用第10个斐波那契数列的值，只要用`Fib<10>::value`就可以了。值得一提的是，由于计算过程需要在编译时完成，模板中的参数必须得是编译时就知晓其值的常量。

另外，通常编译器会对模板的递归层数作限制，在`clang`编译器上默认是256层。可以使用`-ftemplate-depth=N`来将层数设为`N`，但太大的层数会使得编译器自己栈溢出……所以这玩意并没有什么○用。

### 但是这又有什么○用呢？

虽说是节省了运行时间，但必须在编译时确定所有数值，加之`N`还不能太大，感觉上并没有什么○用。

但TML可不光是编译时算数这么简单。下面将介绍两个TML在实际中的应用，请坐和放宽，准备打开新世界的大门。

## CRTP

这个看上去超级高大上的缩写，全称其实是 Curiously Recurring Template Pattern[^1]，直译过来就是“神奇递归模板模式”，一下就显得傻里傻气的了。

为什么叫这个名字呢？我们先来看一段代码：

``` c++
template <class Derived>
struct Base {
	void polymorphism() {
		static_cast<Derived*>(this)->_poly();
	}
	static void static_poly() {
		Derived::_static_poly();
	}
	// Default implementation
	void _poly();
	static _static_poly();
} ;

struct Derived : public Base<Derived> {
	void _poly();
//	void _static_poly();
}
```

嗯，`Base`类是一个正常的模板类，虽然内容有点不明觉厉。但`Derived`类是什么鬼？为啥能把自己作为父类的模板参数？这岂不是“我依赖于我爸依赖于我”的状况 = = ？

然而这段代码是可以通过编译的合法代码。原因有二：

- 虽然`Base`需要了解`Derived`的定义，但其不直接或间接包含`Derived`类的实例，也即其大小不依赖于`Derived`；
- 模板类会在被使用时实例化，而此时`Base`类与`Derived`类的定义均已知晓；就此段代码而言，编译器可以检验`Derived`类是否包含`_poly()`和`_static_poly()`函数的定义，如果没有找到，则会在其父类`Base<Derived>`类中寻找。

一般人在第一次看到这个玄学一般的用法时都会目瞪口呆不知所措。先不要惊讶，我们来看看用这种写法能做什么：

### 无需 VTABLE 的编译时多态

C++ 实现多态的方法是将类内所有的虚（virtual）函数存在一个被称为`VTABLE`的数组中，在运行时调用一个虚函数时，实际调用表中指向的函数。这个方法有个缺点，就是必须维护这么一个表，从而产生额外开销（overhead）。

而利用 CRTP 则可以在实现多态的同时，省去这个表的开销。我们看上面那段代码，在`Base`类（下称基类）和`Derived`类（下称派生类）中都定义有`_poly()`（非虚）函数，理论上基类是访问不到派生类中的函数的。但是这里不一样，基类拥有额外的信息：派生类叫什么。所以基类在调用之前，将自己转换成了派生类。这个转换是可行的，因为事实上，自己本身就是自己的派生类（有点绕，感受一下）。这么一来，调用的就是派生类中的函数了。如果派生类中没有定义`_poly()`函数，则编译器会找到基类中的同名函数；如果定义了`_poly()`函数，则它会覆盖掉基类中的同名函数，故编译器会找到派生类中的函数。这样我们就在编译器实现了多态。

不过 CRTP 的这个多态并非真正的多态。如果我们有一个派生类的实例，通过 CRTP，基类中定义的函数可以调用派生类中重载的“虚”函数。但如果我们的派生类以基类指针的形式存在，我们则无法通过其访问到派生类的虚函数。后者被成为运行时多态，是 CRTP 力不能及的。

### 通用基类

有时候我们会遇到这样的问题：我们只有一个基类的指针，而我们要进行一次深拷贝，如下：

``` c++
struct Base {};
struct Derived : public Base {};
struct AnotherDerived : public Base {};

void copy(Base *p) {
	Base *copy_p = new ???(*p);
}
```

于是我们懵逼了，这个`???`该填啥啊，写基类不对吧，写派生类也不知道是哪个啊，咋办？

一个解决方式定义一个名为`clone()`的虚函数，然后在每个派生类中重载：

``` c++
struct Base {
	virtual Base *clone() const = 0;
} ;
struct Derived : public Base {
	virtual Derived *clone() const {
		return new Derived(*this);
	}
}
struct AnotherDerived : public Base {
	virtual AnotherDerived *clone() const {
		return new AnotherDerived(*this);
	}
}
```

这份代码可以正常运作，但显得十分不优美：每个类里都得写一遍，这样不仅有大量重复代码，还容易出错。

我们来分析一下为什么需要这样写，问题似乎在于，我的基类不知道我到底是啥，所以我需要在派生类里定义虚函数。这不就是 CRTP 解决的问题吗？我们把代码改成下面这样[^2]：

``` c++
struct Base {
	virtual Base *clone() const = 0;
};

template <class Derived>
struct BaseCRTP : public Base {
	virtual Base *clone() const {
		return new Derived(static_cast<Derived const &>(*this));
	}
}

struct Derived : public BaseCRTP<Derived> {};
```

假设我们有一个`Derived`类的实例，在经历了一些事情之后，它变成了`Base`类的指针。现在我们通过这个指针调用`clone()`函数，通过虚函数的机制，我们会找到`BaseCRTP<Derived>`类的`clone()`函数。此时我们已经知道了派生类的名字，也就可以完成深拷贝了。

如果分析一下背后的原因，我们会发现，每次定义一个派生类，编译器都会实例化一份`BaseCRTP`出来。所以并没有什么神奇的，其实只是编译器帮我们生成了我们本应手写的代码而已。CRTP 的其他应用，比如实例计数器，也都是基于这个原理。

## SFINAE

又是一个高大上的缩写，不急，我们先把它展开了：全称为 Substitution failure is not an error[^3]。这都不是一个词组了，而是一句话：替换失败不被视为编译错误。当替换失败时，编译器不报错，而只是将这个模板从待选的重载函数集中移除，不考虑失败的这一个模板而已。

在实例化一个模板时，编译器需要把模板参数中的东西替换为实际的类型或值，然而替换是可能失败的，比如下面这个例子：

``` c++
template <class T>
void defined_foo(typename T::foo) {}

template <class T>
void defined_foo(T) {}

struct Foo {
	typedef int foo;
} ;
```

这里我们定义了两个版本的`defined_foo()`，接受不同的参数。`typename`关键字是为了消除歧义，告诉编译器`T::foo`绝壁是个类型名。由于是模板函数，编译器得做类型替换，这里就可能出现替换失败的情况，比如：

``` c++
defined_foo<Foo>(0);
defined_foo<int>(0);
```

第一行，`Foo`类中有`foo`类型，所以在第一个模板中替换成功，而在第二个模板中替换失败；第二行，`int`类型显然不包含别的类型，所以在第一个模板中替换失败，而在第二个模板中替换成功。这两个情况中，都是恰有一个重载的模板替换成功，都不会报错。

看上去是非常自然而且简单的做法吧？你绝对想不到可以用这玩意儿做什么：

### 编译时自省

先说说什么是自省。子曰：“见贤思齐焉，见不贤而内*自省*也”，不过这和我们要讲的自省没有半毛钱关系。

所谓自省，其实就是程序自己知道自己的情况。比如在 Python 中，程序可以在运行时查看自己的代码，获取某个类的名称、成员，甚至是实时增减成员。对于 C++ 这种强类型语言来说，这显然不现实。但通过 SFINAE，我们可以实现一定程度上的自省。

上面的那个例子，其实就是某种自省。我们可以在编译时判断一个类是否满足某种条件，从而调用不同的函数。在 STL 中，有一个神秘的头文件叫`<type_traits>`（C++11 的新特性），其中包含了很多判断类型的类型的东西，如`is_array`、`is_class`等。这些东西其实是用 SFINAE 实现的模板类，其中包含一个名为`value`的成员，代表判断的值是真是假。

这么说可能有些玄，我们来看一个具体的、真实存在的例子：

### boost::enable_if

先上代码再解释：

``` c++
template<bool Cond, class T = void>
struct enable_if_c
{ typedef T type; };

template<class T>
struct enable_if_c<false, T> {};

template<class Cond, class T = void>
struct enable_if : enable_if_c<Cond::value, T> {};

// === 我是分割线 ===
template<class T>
typename enable_if<is_floating_point<T>, T>::type
  frobnicate(T t) {}
```

`is_floating_point`也是`<type_traits>`里的东西，其作用正如其名。`enable_if`的实现是，如果第一个参数的条件为`true`，那么其中会包含一个`type`类型，为第二个参数的类型；否则不会有这个类型。拿分割线下的函数举例，如果第一个参数的条件，即`T`是浮点类型，为`false`，那么在使用这个函数时就会产生替换错误。所以`enable_if`的作用可以理解为限制模板所能够接受的类型。

不过这个写法有点丑陋，毕竟返回类型这么长；所以另一个常见的写法是添加一个虚设的（dummy）参数，并在参数里实现 SFINAE[^4]：

``` c++
template <class T>
T frobnicate(T t, typename enable_if<is_floating_point<T>, T>::type * = 0) {}
```

### type_traits

整个`<type_traits>`库过于庞大，以个人之力无法看穿，只能挑选一些浅薄的所见所得与大家分享。下面的代码均来自`<type_traits>`库，为了便于阅读，删去了一部分不影响理解的内容。

首先，我们需要两个类`true_type`和`false_type`，用来区分结果。这两个类中应当定义`value`，类型为`bool`，值分别为`true`和`false`。

然后，我们要考虑 cv 修饰符（`const`和`volatile`）的问题，它们不应影响我们对类型的判断。`remove_cv`的作用是去除类型中的 cv 修饰符，它的实现如下：

``` c++
// remove_const
template <class _Tp> struct remove_const {typedef _Tp type;};
template <class _Tp> struct remove_const<const _Tp> {typedef _Tp type;};

// remove_volatile
// ...

// remove_cv
template <class _Tp> struct remove_cv {
	typedef typename remove_volatile<typename remove_const<_Tp>::type>::type type;
};
```

这里利用模板的匹配功能，获得了类型去掉 cv 修饰符后的名字，并记在了`remove_cv::type`中。

接下来就可以判断了。先看一个简单的，`is_null_pointer`：

``` c++
template <class _Tp> struct __is_nullptr_t_impl : public false_type {};
template <> struct __is_nullptr_t_impl<nullptr_t> : public true_type {};

template <class _Tp> struct _LIBCPP_TYPE_VIS_ONLY is_null_pointer
    : public __is_nullptr_t_impl<typename remove_cv<_Tp>::type> {};
```

`nullptr_t`是 C++11 中加入的空指针类型。这个判断很简单，不用多说。

下面则是一个特别玄的判断：`is_base_of`，判断`B`是否是`D`的基类。由于`<type_traits>`中的代码过于复杂，下面给出的代码是 StackOverflow 上一个问题里[^5]的简化版代码：

``` c++
typedef char (&yes)[1];
typedef char (&no)[2];

template <class B, class D>
struct Host {
	operator B*() const;
	operator D*();
} ;

template <class B, class D>
struct is_base_of {
	template <class T> 
	  static yes test(D*, T);
	static no test(B*, int);
	
	static const bool value = sizeof(test(Host<B,D>(), int())) == sizeof(yes);
} ;
```

我们把`true_type`和`false_type`改成了`yes`和`no`，通过其内存大小来判断类型；这个并不重要。

要理解这段代码的原理，首先我们需要知道 C++ 标准中，有多个可选函数时会优先选择哪方：

- **原则1：**如果两个函数参数类型相同，而 cv 修饰符不同，则优先选择与传入参数 cv 修饰符匹配的一方；
- **原则2：**如果原则1无法区分，且两个类型转换函数返回类型不同，则优先选择与目标参数匹配的一方；
- **原则3：**如果原则2无法区分，优先选择非模板函数。

现在我们来分析一下代码的原理。`Host`的两个类型转换函数的原型分别为

- `B *(Host<B, D> const &)`
- `D *(Host<B, D> &)`

假设`B`是`D`的基类，那么`D *`可以转换为`B *`，反则不行。对于第一个`test`函数，可选的转换函数只有第二个；而对于第二个`test`函数，两个转换函数都可选，根据原则1，编译器会第二个选择转换函数（因为默认的传入参数为`*this`，为非`const`类型）。此时第一个`test`函数的目标类型与转换函数的返回类型匹配，而第二个的不匹配，根据原则2，编译器会选择第一个`test`函数，故得到`yes`。

假设`B`不是`D`的基类，那么`D *`不可以转换为`B *`，反之或许可以。对于第二个`test`函数，可选的转换函数只有第一个；对于第一个`test`函数，可选的转换函数有第二个，也可能有第一个，但一定会选择第二个。此时根据原则3，编译器会选择第二个`test`函数，故得到`no`。

## 总结

写了这么多，其实也只涉及了模板的冰山一角。由此可见这一功能的强大与复杂，也不难理解为何模板一直处于争论的中心，甚至有这么一个笑话：“Java 程序员聚在一起谈面向对象和设计模式，C++ 程序员聚在一起谈模板和语言规范到底是怎么回事”，嘲笑的就是 C++ 令人咂舌的复杂程度。

但一码归一码，C++ 还是一门被广泛使用的语言，因此适当的了解还是必要的。而本文中提到的用法，可能大家没有见过，但在工程中的确是普遍存在的。就拿 CRTP 来说，Boost 库的文法分析库 Spirit、计算几何库 CGAL 的整个核心中都使用了 CRTP；而 SFINAE 更是已经进入了 C++ 标准。即便自己不会写出这样的代码，至少在见到的时候也应该明白是在干什么；退一步讲，把这篇文章当成是普通的科普文章，图个乐呵也好。

## 参考文献

[^1]: [Wikipedia - Curiously recurring template pattern](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
  
[^2]: [Katy’s Code - C++: Polymorphic cloning and the CRTP (Curiously Recurring Template Pattern)](https://katyscode.wordpress.com/2013/08/22/c-polymorphic-cloning-and-the-crtp-curiously-recurring-template-pattern/)
  
[^3]: [Wikipedia - Substitution failure is not an error](https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error)
  
[^4]: [ACCU 2013 - Jonathan Wakely - SFINAE Functionality Is Not Arcane Esoterica](http://accu.org/content/conf2013/Jonathan_Wakely_sfinae.pdf)
  
[^5]: [StackOverflow - How does `is_base_of` work?](http://stackoverflow.com/questions/2910979/how-does-is-base-of-work)
  
  ​
